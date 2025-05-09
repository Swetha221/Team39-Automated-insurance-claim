import os
import requests
import uuid
import pyodbc
import json
import traceback
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()
app = Flask(__name__)

# Azure Key Vault
KEY_VAULT_NAME = os.environ.get("KEY_VAULT_NAME")
KV_URL = f"https://{KEY_VAULT_NAME}.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=KV_URL, credential=credential)

# Load secrets
VISION_KEY = secret_client.get_secret("COMPUTER-VISION-KEY").value
VISION_ENDPOINT = secret_client.get_secret("COMPUTER-VISION-ENDPOINT").value
FORM_RECOGNIZER_KEY = secret_client.get_secret("FORM-RECOGNIZER-KEY").value
FORM_RECOGNIZER_ENDPOINT = secret_client.get_secret("FORM-RECOGNIZER-ENDPOINT").value
COSMOS_DB_URI = secret_client.get_secret("COSMOS-ENDPOINT").value
COSMOS_DB_KEY = secret_client.get_secret("COSMOS-KEY").value
COSMOS_DB_DATABASE = secret_client.get_secret("COSMOS-DB-NAME").value
COSMOS_DB_CONTAINER = secret_client.get_secret("COSMOS-CONTAINER-NAME").value
GPT_API_KEY = secret_client.get_secret("AZURE-OPENAI-KEY").value
AZURE_OPENAI_ENDPOINT = secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value
BLOB_CONNECTION_STRING = secret_client.get_secret("BLOB-CONNECTION-STRING").value
BLOB_CONTAINER_NAME = secret_client.get_secret("BLOB-CONTAINER-NAME").value
SQL_SERVER = secret_client.get_secret("SQL-SERVER").value
SQL_DATABASE = secret_client.get_secret("SQL-DATABASE").value
SQL_USERNAME = secret_client.get_secret("SQL-USERNAME").value
SQL_PASSWORD = secret_client.get_secret("SQL-PASSWORD").value
LOGIC_APP_WEBHOOK_URL = secret_client.get_secret("LOGIC-APP-WEBHOOK-URL").value

# Azure clients
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
form_recognizer = DocumentAnalysisClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_KEY))
vision_client = ComputerVisionClient(VISION_ENDPOINT, CognitiveServicesCredentials(VISION_KEY))
cosmos_client = CosmosClient(COSMOS_DB_URI, credential=COSMOS_DB_KEY)
container = cosmos_client.get_database_client(COSMOS_DB_DATABASE).get_container_client(COSMOS_DB_CONTAINER)
openai_client = AzureOpenAI(api_key=GPT_API_KEY, api_version="2024-02-15-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT)

# SQL connection
sql_connection_string = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};UID={SQL_USERNAME};PWD={SQL_PASSWORD}"
)
try:
    sql_conn = pyodbc.connect(sql_connection_string)
    print("✅ Successfully connected to SQL Server.")
except Exception as e:
    print("❌ SQL connection failed:", e)

def validate_claim(name, email):
    try:
        cursor = sql_conn.cursor()
        cursor.execute("""
            SELECT c.CustomerID, p.PolicyID
            FROM Customers c
            JOIN Policies p ON c.CustomerID = p.CustomerID
            WHERE LOWER(c.Name) = LOWER(?) AND LOWER(c.Email) = LOWER(?) AND p.Status = 'Active'
        """, (name, email))
        result = cursor.fetchone()
        if result:
            return {"isValid": True, "customerId": result[0], "policyId": result[1]}
    except Exception as e:
        print("❌ SQL validation error:", str(e))
    return {"isValid": False}

@app.route("/", methods=["GET"])
def serve_form():
    return render_template("index.html")

@app.route("/submit-claim", methods=["POST"])
def submit_claim():
    try:
        name = request.form.get("name")
        email = request.form.get("email")
        description = request.form.get("accidentDescription")
        accident_date = request.form.get("accidentDate")
        vehicle_model = request.form.get("vehicleModel")
        files = request.files.getlist("carPhotos") + request.files.getlist("supportingDocuments")

        if not files or all(f.filename == "" for f in files):
            return jsonify({"status": "error", "message": "No files uploaded."}), 400

        validation = validate_claim(name, email)
        if not validation["isValid"]:
            return jsonify({"status": "error", "message": "User does not have a valid policy."}), 400

        customer_id = validation["customerId"]
        policy_id = validation["policyId"]
        claim_id = str(uuid.uuid4())
        document_details = []

        for file in files:
            filename = secure_filename(file.filename)
            blob_client = container_client.get_blob_client(filename)
            blob_client.upload_blob(file.stream, overwrite=True)
            blob_url = blob_client.url
            sas_token = generate_blob_sas(
                account_name=blob_service_client.account_name,
                container_name=BLOB_CONTAINER_NAME,
                blob_name=filename,
                account_key=blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(minutes=15)
            )
            blob_url_with_sas = f"{blob_url}?{sas_token}"

            caption = "No caption detected"
            try:
                vision_result = vision_client.describe_image(blob_url_with_sas, max_candidates=1)
                if vision_result.captions:
                    caption = vision_result.captions[0].text
            except Exception as e:
                print("❌ Computer Vision failed:", e)

            form_data = {}
            try:
                poller = form_recognizer.begin_analyze_document_from_url("prebuilt-document", blob_url_with_sas)
                result = poller.result()
                for kv in result.key_value_pairs:
                    key = kv.key.content.strip() if kv.key and kv.key.content else ""
                    value = kv.value.content.strip() if kv.value and kv.value.content else ""
                    if key:
                        form_data[key] = value
            except Exception as e:
                form_data = {"error": f"Recognizer error: {str(e)}"}
                print("❌ Form Recognizer failed:", form_data)

            try:
                cursor = sql_conn.cursor()
                cursor.execute("""
                    INSERT INTO Documents (ClaimID, DocumentType, FileName, FileUrl)
                    VALUES (?, ?, ?, ?)
                """, (claim_id, "Photo", filename, blob_url_with_sas))
                sql_conn.commit()
            except Exception as e:
                print("❌ SQL insert error for document:", str(e))

            document_details.append({
                "file": filename,
                "caption": caption,
                "formData": form_data,
                "blobUrl": blob_url_with_sas
            })

        # GPT-4 summary analysis
        gpt_summary = "Summary not available due to error."
        gpt_prompt = (
            f"Summarize the following vehicle accident report clearly and concisely so it can be cross-validated with visual damage evidence:\n\n"
            f"{description}"
        )
        try:
            gpt_response = openai_client.chat.completions.create(
                model="gpt-4o-39",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant summarizing vehicle accident claims."},
                    {"role": "user", "content": gpt_prompt}
                ],
                temperature=0.3,
            )
            gpt_summary = gpt_response.choices[0].message.content.strip()
            print("🧠 GPT Summary:", gpt_summary)
        except Exception as e:
            print("❌ GPT summary error:", str(e))
            traceback.print_exc()

        claim_data = {
            "id": claim_id,
            "name": name,
            "email": email,
            "customerId": customer_id,
            "policyId": policy_id,
            "accidentDate": accident_date,
            "vehicleModel": vehicle_model,
            "description": description,
            "gptSummary": gpt_summary,
            "documents": document_details,
            "status": "Submitted"
        }

        container.upsert_item(claim_data)
        print("✅ Claim stored in Cosmos DB.")

        try:
            if LOGIC_APP_WEBHOOK_URL:
                requests.post(LOGIC_APP_WEBHOOK_URL, json=claim_data)
        except Exception as e:
            print("❌ Logic App webhook failed:", str(e))

        return render_template("success.html", message="Claim submitted successfully.", claim_id=claim_id)

    except Exception as e:
        print("❌ General error in submit_claim:", str(e))
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
