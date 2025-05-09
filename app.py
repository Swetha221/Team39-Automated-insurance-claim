import os
import requests
import uuid
import pyodbc
from flask import Flask, request, jsonify, render_template, redirect, url_for
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

KEY_VAULT_NAME = os.environ.get("KEY_VAULT_NAME")
KV_URL = f"https://{KEY_VAULT_NAME}.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=KV_URL, credential=credential)

AZURE_CLIENT_ID = secret_client.get_secret("AZURE-CLIENT-ID").value
AZURE_TENANT_ID = secret_client.get_secret("AZURE-TENANT-ID").value
AZURE_CLIENT_SECRETS = secret_client.get_secret("AZURE-CLIENT-SECRETS").value
os.environ["AZURE-CLIENT-ID"] = AZURE_CLIENT_ID
os.environ["AZURE-TENANT-ID"] = AZURE_TENANT_ID
os.environ["AZURE-CLIENT-SECRETS"] = AZURE_CLIENT_SECRETS

VISION_KEY = secret_client.get_secret("COMPUTER-VISION-KEY").value
VISION_ENDPOINT = secret_client.get_secret("COMPUTER-VISION-ENDPOINT").value
FORM_RECOGNIZER_KEY = secret_client.get_secret("FORM-RECOGNIZER-KEY").value
FORM_RECOGNIZER_ENDPOINT = secret_client.get_secret("FORM-RECOGNIZER-ENDPOINT").value
COSMOS_DB_URI = secret_client.get_secret("COSMOS-ENDPOINT").value
COSMOS_DB_KEY = secret_client.get_secret("COSMOS-KEY").value
COSMOS_DB_DATABASE = secret_client.get_secret("COSMOS-DB-NAME").value
COSMOS_DB_CONTAINER = secret_client.get_secret("COSMOS-CONTAINER-NAME").value
LOGIC_APP_WEBHOOK_URL = secret_client.get_secret("LOGIC-APP-WEBHOOK-URL").value
GPT_API_KEY = secret_client.get_secret("AZURE-OPENAI-KEY").value
AZURE_OPENAI_ENDPOINT = secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value
BLOB_CONNECTION_STRING = secret_client.get_secret("BLOB-CONNECTION-STRING").value
BLOB_CONTAINER_NAME = secret_client.get_secret("BLOB-CONTAINER-NAME").value

SQL_SERVER = secret_client.get_secret("SQL-SERVER").value
SQL_DATABASE = secret_client.get_secret("SQL-DATABASE").value
SQL_USERNAME = secret_client.get_secret("SQL-USERNAME").value
SQL_PASSWORD = secret_client.get_secret("SQL-PASSWORD").value

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

sql_connection_string = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};"
    f"UID={SQL_USERNAME};PWD={SQL_PASSWORD}"
)
try:
    sql_conn = pyodbc.connect(sql_connection_string)
    print("✅ Successfully connected to SQL Server.")
except Exception as e:
    print("❌ SQL connection failed:", e)

form_recognizer = DocumentAnalysisClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_KEY))
vision_client = ComputerVisionClient(VISION_ENDPOINT, CognitiveServicesCredentials(VISION_KEY))
cosmos_client = CosmosClient(COSMOS_DB_URI, credential=COSMOS_DB_KEY)
container = cosmos_client.get_database_client(COSMOS_DB_DATABASE).get_container_client(COSMOS_DB_CONTAINER)
openai_client = AzureOpenAI(api_key=GPT_API_KEY, api_version="2024-02-15-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT)

def validate_claim(name, email):
    try:
        cursor = sql_conn.cursor()
        cursor.execute("""
            SELECT c.CustomerID, p.PolicyID
            FROM Customers c
            JOIN Policies p ON c.CustomerID = p.CustomerID
            WHERE c.Name = ? AND c.Email = ? AND p.Status = 'Active'
        """, (name, email))
        result = cursor.fetchone()
        if result:
            return {"isValid": True, "customerId": result[0], "policyId": result[1]}
        else:
            return {"isValid": False}
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
        policy_id = request.form.get("policyId")
        accident_date = request.form.get("accidentDate")
        vehicle_model = request.form.get("vehicleModel")
        car_photos = request.files.getlist("carPhotos")
        supporting_docs = request.files.getlist("supportingDocuments")
        files = car_photos + supporting_docs

        if not files or all(f.filename == "" for f in files):
            return jsonify({"status": "error", "message": "No files uploaded."}), 400

        validation_result = validate_claim(name, email)
        if not validation_result["isValid"]:
            return jsonify({"status": "error", "message": "User does not have a valid policy."}), 400

        customer_id = validation_result["customerId"]
        policy_id = validation_result["policyId"]
        claim_id = str(uuid.uuid4())
        document_details = []

        for file in files:
            if file:
                form_data_dict = {}
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
                    expiry=datetime.now(tz=timezone.utc) + timedelta(minutes=15)
                )
                blob_url_with_sas = f"{blob_url}?{sas_token}"
                print(f"✅ Uploaded to blob with SAS: {blob_url_with_sas}")

                try:
                    vision_result = vision_client.describe_image(blob_url_with_sas, max_candidates=1)
                    caption = vision_result.captions[0].text if vision_result.captions else "No caption detected"
                except Exception as e:
                    caption = f"Vision error: {str(e)}"
                    print("❌ Computer Vision failed:", caption)

                try:
                    poller = form_recognizer.begin_analyze_document_from_url("prebuilt-document", blob_url_with_sas)
                    result = poller.result()
                    form_data_dict = {}
                    for kv in result.key_value_pairs:
                        key = kv.key.content.strip() if kv.key and kv.key.content else ""
                        value = kv.value.content.strip() if kv.value and kv.value.content else ""
                        if key:
                            form_data_dict[key] = value
                except Exception as e:
                    form_data_dict = {"error": f"Recognizer error: {str(e)}"}
                    print("❌ Form Recognizer failed:", form_data_dict)

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
					"formData": form_data_dict, 
					"blobUrl": blob_url_with_sas
				})

        claim_data = {
            "id": claim_id,
            "name": name,
            "email": email,
            "customerId": customer_id,
            "policyId": policy_id,
            "accidentDate": accident_date,
            "vehicleModel": vehicle_model,
            "description": description,
            "documents": document_details,
            "status": "Submitted"
        }

        try:
            container.upsert_item(claim_data)
            print("✅ Claim stored in Cosmos DB.")
        except Exception as e:
            print("❌ Cosmos DB write failed:", str(e))

        try:
            if LOGIC_APP_WEBHOOK_URL:
                requests.post(LOGIC_APP_WEBHOOK_URL, json=claim_data)
        except Exception as e:
            print("❌ Logic App webhook failed:", str(e))

        return render_template("success.html", message="Claim submitted successfully.", claim_id=claim_id)

    except Exception as e:
        print("❌ General error in submit_claim:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
