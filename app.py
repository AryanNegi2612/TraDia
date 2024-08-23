
import streamlit as st
import os
from pydub import AudioSegment
import speech_recognition as sr
import noisereduce as nr

# Streamlit app title
st.title("Speech-to-Text Application")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_name = uploaded_file.name
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert to .wav format if necessary
    if file_name.endswith(".mp3"):
        audio = AudioSegment.from_mp3(file_name)
        file_name = file_name.replace(".mp3", ".wav")
        audio.export(file_name, format="wav")

    # Load the audio file for speech recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio_data = recognizer.record(source)

    # Noise reduction (optional)
    st.write("Applying noise reduction...")
    reduced_noise = nr.reduce_noise(y=audio_data.get_array_of_samples(), sr=audio_data.sample_rate)
    
    # Recognize speech
    st.write("Converting speech to text...")
    try:
        text = recognizer.recognize_google(sr.AudioData(reduced_noise.tobytes(), audio_data.sample_rate, audio_data.sample_width))
        st.text_area("Transcribed Text:", text)
    except sr.UnknownValueError:
        st.write("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
