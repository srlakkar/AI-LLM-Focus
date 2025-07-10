import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_ollama(prompt, model="llama3"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

if __name__ == "__main__":
    user_prompt = "Explain how a black hole forms in simple terms."
    response = ask_ollama(user_prompt)
    print("Ollama says:\n", response)
