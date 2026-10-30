
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("NVIDIA_API_KEY")
base_url = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
model = os.getenv("MODEL_NAME", "z-ai/glm-5.3-flash")

if not api_key:
    raise RuntimeError("NVIDIA_API_KEY is missing from .env")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=120.0,
    max_retries=0,
)

print(f"Testing model: {model}")
start = time.time()

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Explain in two sentences why an unusually high "
                    "healthcare claim amount may need investigation."
                ),
            }
        ],
        temperature=0.2,
        max_tokens=150,
    )

    print("\nSUCCESS")
    print(f"Elapsed seconds: {time.time() - start:.1f}")
    print(response.choices[0].message.content)

except Exception as exc:
    print("\nFAILED")
    print(f"Elapsed seconds: {time.time() - start:.1f}")
    print(f"Error type: {type(exc).__name__}")
    print(f"Error details: {exc}")
