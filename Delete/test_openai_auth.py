# test_openai_auth.py
from dotenv import load_dotenv
import os
from openai import OpenAI

# Force .env to override machine/user env vars
load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT"),
    # organization=os.getenv("OPENAI_ORG_ID"),  # optional; omit during auth debug
)

print("Using key:", (os.getenv("OPENAI_API_KEY") or "")[:12] + "…")
print("Project:", os.getenv("OPENAI_PROJECT"))
print("Org:", os.getenv("OPENAI_ORG_ID") or "<omitted>")

try:
    models = client.models.list()
    ids = [m.id for m in models.data]
    print("OK: model list retrieved (showing first 10):")
    for mid in ids[:10]:
        print(" •", mid)
    print("\nCheck that 'gpt-4o-mini' appears above.")
except Exception as e:
    print("ERROR talking to API:")
    print(repr(e))
