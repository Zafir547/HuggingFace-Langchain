import os
import base64
from PIL import Image
import requests
import streamlit as st
from io import BytesIO
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
from IPython.display import Audio

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load image-to-text pipeline only once
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# img2text
def img2text(image_bytes):
    # Open the image using PIL
    img = Image.open(BytesIO(image_bytes))
    # Convert the image to RGB mode (required for the pipeline)
    img = img.convert("RGB")
    # Save the image to a temporary file
    temp_image_path = "temp_image.jpg"
    img.save(temp_image_path)

    # Use the pipeline with the temporary image path
    text = image_to_text(temp_image_path)[0]["generated_text"]
    
    # Clean up - remove the temporary image file
    os.remove(temp_image_path)
    
    print(text)
    return text

# llm
def generate_story(scenario):
    template = """
    You are a storyteller;
    You can generate a short story based on a single narrative, 
    the story should be no more than 20 words;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    print(story)
    return story

# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}
    payloads = {"input": str(message)}  # Ensure message is a string

    try:
        response = requests.post(API_URL, headers=headers, data=payloads)
        response.raise_for_status()  # Check for errors in the response

        with open('audio1.flac', 'wb') as file:
            file.write(response.content)
        file.close()  # Close the file after writing audio

    except requests.exceptions.RequestException as e:
        print(f"Error during text-to-speech API request: {e}")

        if response.status_code == 400:
            print("Bad Request. API response content:")
            print(response.content)
        elif response.status_code == 403:
            print("Unauthorized. Check your API token.")
        elif response.status_code == 404:
            print("API endpoint not found.")
        else:
            print(f"Unexpected error. API response status code: {response.status_code}")

        return None

    return response.content

def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="🎂")
    st.header("Turn img into an audio story")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()  # Use read() instead of getvalue()
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        scenario = img2text(bytes_data)
        story = generate_story(scenario)
        audio_content = text2speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        if audio_content:
            st.audio(audio_content, format="audio/flac")

if __name__ == '__main__':
    main()
