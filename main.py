from flask import Flask, request, jsonify
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
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
        self.client_secret_file = os.getenv("CLIENT_SECRET_FILE")
        self.model_name = os.getenv("MODEL_NAME")
        self.model_revision = os.getenv("MODEL_REVISION")
        self.creds = None
        self.service = None

        # Initialize sentiment analysis pipeline
        self.model = pipeline("sentiment-analysis", model=self.model_name, revision=self.model_revision)

        # Authenticate
        self.authenticate_user()

    def authenticate_user(self):
        """Authenticate user using OAuth 2.0."""
        flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_file, self.SCOPES)
        self.creds = flow.run_local_server(port=0)
        self.service = build("chat", "v1", credentials=self.creds)
        print("✅ Authentication successful.")

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

    def get_messages(self, space_id, hours_ago=1):
        """Retrieve and analyze messages from Google Chat Space."""
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

    hours_ago = int(request.args.get("hours_ago", 1))
    messages = detector.get_messages(space_id, hours_ago)
    return jsonify(messages)


if __name__ == "__main__":
    from os import environ

    app.run(host="0.0.0.0", port=int(environ.get("PORT", 5000)), debug=True)
