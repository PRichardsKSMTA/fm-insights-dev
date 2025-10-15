from dotenv import load_dotenv
import os

# load values from .env
load_dotenv()

print("KEY prefix:", (os.getenv("OPENAI_API_KEY") or "")[:12])
print("PROJECT:", os.getenv("OPENAI_PROJECT"))
print("ORG:", os.getenv("OPENAI_ORG_ID"))
