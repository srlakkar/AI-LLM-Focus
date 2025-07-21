import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import ollama

st.set_page_config(page_title="Image Description with LLaVA", layout="centered")

st.title("🖼️ Image Description using Ollama + LLaVA")

# Input: Image URL
image_url = st.text_input("Enter Image URL:", 
                          "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg")

if st.button("Generate Description"):
    try:
        # Download image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Display the image
        st.image(image, caption="Input Image", use_column_width=True)

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Prompt
        prompt = "Describe this image in detail."

        # Call Ollama with LLaVA model
        with st.spinner("Generating description..."):
            response = ollama.chat(
                model="llava:7b",
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_base64]
                }]
            )

        description = response['message']['content']
        st.success("**Description:**")
        st.write(description)

    except Exception as e:
        st.error(f"Error: {e}")
