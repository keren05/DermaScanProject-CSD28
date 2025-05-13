import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import time
from geopy.geocoders import Nominatim
import requests
from openai import OpenAI
from pprint import pformat
import openai
from assistant import ASSISTANT_ID
# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set in secrets
client = openai.Client()

# Use ASSISTANT_ID in your app logic
print(f"Using Assistant ID: {ASSISTANT_ID}")
# Run this once to get ASSISTANT_ID

# Streamlit UI
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🩺",
    layout="wide"
)



# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('isic_skin_classifier.h5')


model = load_model()

# Disease database with climate considerations
disease_db = {
    'Melanoma': {
        'diagnosis': ["Asymmetrical shape", "Irregular borders", "Color variation", "Diameter >6mm"],
        'treatment': {
            'global': ["Surgical excision", "Immunotherapy"],
            'hot': ["Increased sun protection", "Morning treatment schedules"],
            'cold': ["Monitor vitamin D levels"]
        },
        'urgency': "High"
    },
    'Melanocytic nevus': {
        'diagnosis': ["Symmetrical", "Smooth borders", "Uniform color"],
        'treatment': {
            'global': ["Observation", "Dermoscopic monitoring"],
            'hot': ["Annual checks", "SPF 50+"],
            'cold': ["Moisturize regularly"]
        },
        'urgency': "Low"
    }
}

# Class names (update with your actual classes)
class_names = [
    'Melanoma',
    'Melanocytic nevus',
    'Basal cell carcinoma',
    'Actinic keratosis',
    'Benign keratosis',
    'Dermatofibroma',
    'Vascular lesion'
]

# Country climate data (simplified)
country_climate = {
    'India': {'climate': 'hot', 'recommendations': ['Avoid midday sun', 'Use broad-spectrum sunscreen']},
    'Canada': {'climate': 'cold', 'recommendations': ['Monitor seasonal changes', 'Winter skin protection']},
    'UK': {'climate': 'temperate', 'recommendations': ['Year-round SPF 30+']}
}


# Image analysis
def predict(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    pred = model.predict(np.expand_dims(img_array, axis=0))
    pred_class = class_names[np.argmax(pred)]
    confidence = round(100 * np.max(pred), 2)
    return pred_class, confidence


# Get climate-specific treatment
def get_climate_treatment(disease, country):
    climate = country_climate.get(country, {}).get('climate', 'global')
    treatments = disease_db[disease]['treatment']
    return treatments.get(climate, treatments['global']) + country_climate.get(country, {}).get('recommendations', [])


# OpenAI Assistant integration
def get_ai_response(thread_id, user_input, diagnosis_data=None):
    # Create message with clinical context
    content = f"Patient from {diagnosis_data['country']} with {diagnosis_data['skin_type']} skin and {diagnosis_data['outdoor_hours']}hr/day sun exposure asks:\n{user_input}\n\n"
    content += f"Diagnosis: {diagnosis_data['pred_class']} ({diagnosis_data['confidence']}% confidence)\n"
    content += f"Climate: {country_climate.get(diagnosis_data['country'], {}).get('climate', 'unknown')}\n"

    # Create or use existing thread
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id

    # Add message to thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )

    # Run assistant
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )

    # Wait for completion
    while run.status != "completed":
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )

    # Get response
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return messages.data[0].content[0].text.value, thread_id

def main():
    st.title("DermaScan AI 🩺")
    st.heading("DermaScan AI 🩺")
    st.write("Skin disease detection with diagnosis and treatment guidance")

    # Sidebar
    with st.sidebar:
        st.header("Input Options")
        option = st.radio("Select input method:",
                          ["Upload Image", "Take Photo", "View Samples"])

        st.divider()
        st.write("**About**")
        st.caption("This tool provides preliminary analysis only. Always consult a dermatologist for medical diagnosis.")

# Input Section
col1, col2 = st.columns(2)
with col1:
    st.header("Image Analysis")
    option = st.radio("Select input method:", ["Upload Image", "Take Photo", "Use Sample"])

    image = None
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose skin image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    elif option == "Take Photo":
        if st.button("Open Camera"):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    elif option == "Use Sample":
        sample_class = st.selectbox("Select sample type:", list(disease_db.keys()))
        if st.button("Load Sample"):
            try:
                image = Image.open(f"samples/{sample_class.lower().replace(' ', '_')}.jpg")
            except FileNotFoundError:
                st.warning(f"Sample image not found for {sample_class}. Using generic example.")
                # Create a blank image as fallback
                image = Image.new('RGB', (224, 224), color=(73, 109, 137))
                st.image(image, caption="Generic Sample Image")

with col2:
    st.header("Patient Context")
    country = st.selectbox("Your Country:", list(country_climate.keys()))
    skin_type = st.select_slider("Skin Type:",
                                 ["I (Very fair)", "II (Fair)", "III (Medium)", "IV (Olve)", "V (Brown)", "VI (Dark)"])
    outdoor_hours = st.slider("Daily outdoor exposure (hours):", 0, 12, 2)

# Initialize session state for OpenAI thread
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Analysis Section
if image and st.button("Analyze"):
    with st.spinner("Diagnosing..."):
        st.image(image, width=300)
        pred_class, confidence = predict(image)

        # Store diagnosis in session state
        st.session_state.diagnosis_data = {
            "pred_class": pred_class,
            "confidence": confidence,
            "country": country,
            "skin_type": skin_type,
            "outdoor_hours": outdoor_hours
        }

        st.success(f"**Diagnosis:** {pred_class} ({confidence}% confidence)")
        st.warning(f"**Urgency:** {disease_db[pred_class]['urgency']}")

        with st.expander("📋 Diagnostic Criteria"):
            st.write("\n".join([f"- {x}" for x in disease_db[pred_class]['diagnosis']]))

        with st.expander("💊 Treatment Plan"):
            st.subheader("Standard Treatments:")
            st.write("\n".join([f"- {x}" for x in disease_db[pred_class]['treatment']['global']]))

            st.subheader(f"Climate-specific for {country}:")
            climate_tips = get_climate_treatment(pred_class, country)
            st.write("\n".join([f"- {x}" for x in climate_tips]))

            if outdoor_hours > 4:
                st.warning(f"⚠️ Reduce sun exposure (currently {outdoor_hours} hrs/day)")

# Chat Assistant
st.divider()
st.header("💬 Ask the Dermatology Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your diagnosis..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if 'diagnosis_data' in st.session_state:
            # Get AI response with clinical context
            response, thread_id = get_ai_response(
                st.session_state.thread_id,
                prompt,
                st.session_state.diagnosis_data
            )
            st.session_state.thread_id = thread_id
        else:
            response = "Please analyze an image first for personalized advice"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})