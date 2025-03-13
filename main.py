from flask import Flask, request, jsonify
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import re
import nltk
import ssl
import datetime
import os
from nltk.corpus import stopwords
from datetime import datetime, timedelta
from transformers import pipeline
from dateutil.parser import parse
from dotenv import load_dotenv
import base64
import json

# ✅ Load environment variables
load_dotenv()

# ✅ Fix SSL issue when downloading stopwords
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# ✅ Download stopwords (only required once)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

app = Flask(__name__)


class ChatIssueDetector:
    SCOPES = [
        "https://www.googleapis.com/auth/chat.messages.readonly",
        "https://www.googleapis.com/auth/chat.spaces"
    ]

    def __init__(self):
        self.service = None
        self.model_name = os.getenv("MODEL_NAME")
        self.model_revision = os.getenv("MODEL_REVISION")

        # ✅ Decode client_secret.json from env variable and save it
        self.client_secret_file = "client_secret.json"
        self.save_client_secret()

        # ✅ Initialize sentiment analysis pipeline
        self.model = pipeline("sentiment-analysis", model=self.model_name, revision=self.model_revision)

    def save_client_secret(self):
        """Decode base64 client_secret.json and save it to a file."""
        encoded_json = os.getenv("CLIENT_SECRET_JSON")
        if not encoded_json:
            raise ValueError("CLIENT_SECRET_JSON not set in environment variables")

        decoded_json = base64.b64decode(encoded_json).decode("utf-8")
        with open(self.client_secret_file, "w") as f:
            f.write(decoded_json)
        print("✅ client_secret.json saved successfully.")

    def authenticate_user(self, token):
        """Authenticate user using the OAuth token from the Authorization header."""
        if not token:
            raise ValueError("❌ No OAuth token provided in the request.")

        # ✅ Create credentials from token
        self.creds = Credentials(token)
        self.service = build("chat", "v1", credentials=self.creds)
        print("✅ Authentication successful using OAuth token.")

    def detect_issue(self, text):
        """Checks if a message contains issue-related keywords or negative sentiment."""
        cleaned_text = re.sub(r"^/\S+\s*", "", text).strip()
        if not cleaned_text:
            return False

        issue_keywords = ["issue", "problem", "error", "bug", "fail", "crash", "not working"]
        if any(re.search(rf"\b{kw}\b", cleaned_text.lower()) for kw in issue_keywords):
            return True

        try:
            result = self.model(cleaned_text)
            sentiment = result[0]["label"]
            return sentiment.upper() == "NEGATIVE"
        except Exception as e:
            print(f"❌ Sentiment Analysis Error: {e}")
            return False

    def get_messages(self, space_id, token, hours_ago=1):
        """Retrieve and analyze messages from Google Chat Space."""
        # ✅ Authenticate using the provided OAuth token
        self.authenticate_user(token)

        space_name = f"spaces/{space_id}"
        now = datetime.utcnow()
        time_threshold = now - timedelta(hours=hours_ago)
        formatted_time = time_threshold.isoformat("T") + "Z"

        response = self.service.spaces().messages().list(
            parent=space_name, orderBy="createTime desc", pageSize=10
        ).execute()

        messages = []
        for msg in response.get("messages", []):
            text = msg.get("text", "No Text")
            msg_time = msg.get("createTime", "")
            cleaned_text = re.sub(r"^/\S+\s*", "", text).strip()
            msg_time_obj = parse(msg_time)

            if msg_time_obj >= parse(formatted_time):
                messages.append({
                    "message": cleaned_text,
                    "timestamp": msg_time,
                    "is_issue": self.detect_issue(cleaned_text)
                })

        return messages


# ✅ Initialize Detector
detector = ChatIssueDetector()


@app.route("/get_messages", methods=["GET"])
def fetch_messages():
    """API Endpoint to fetch messages from Google Chat."""
    space_id = request.args.get("space_id")
    if not space_id:
        return jsonify({"error": "Missing space_id parameter"}), 400

    # ✅ Extract token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401

    token = auth_header.split("Bearer ")[1]

    hours_ago = int(request.args.get("hours_ago", 1))
    messages = detector.get_messages(space_id, token, hours_ago)
    return jsonify(messages)


if __name__ == "__main__":
    from os import environ

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)

