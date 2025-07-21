import requests
from PIL import Image
from io import BytesIO
import base64
import ollama

def describe_image(image_url):
    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Send request to Ollama with LLaVA model
    prompt = "Describe this image in detail."
    response = ollama.chat(
        model="llava:7b",
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [img_base64]
        }]
    )

    return response['message']['content']

# Example usage
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
description = describe_image(url)
print("Image description:", description)
