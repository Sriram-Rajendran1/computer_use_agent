import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gpt-4o-mini")
VISION_ESCALATION_MODEL = os.environ.get("VISION_ESCALATION_MODEL", "gpt-4o")
COMPUTER_USE_MODEL = os.environ.get("COMPUTER_USE_MODEL", "gpt-5.6-sol")

MAX_VERIFY_RETRIES = int(os.environ.get("MAX_VERIFY_RETRIES", "3"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "50"))
RECALL_LIMIT = int(os.environ.get("RECALL_LIMIT", "5"))
