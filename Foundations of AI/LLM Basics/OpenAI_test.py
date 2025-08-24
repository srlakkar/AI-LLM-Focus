import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_openai(prompt, model="gpt-4"):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    prompt = "What's the difference between AI and machine learning?"
    reply = ask_openai(prompt)
    print("OpenAI says:\n", reply)
