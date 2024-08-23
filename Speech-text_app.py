import streamlit as st
import os
from pydub import AudioSegment
import noisereduce as nr
import librosa
import soundfile as sf
import speech_recognition as sr
from pyannote.audio import Pipeline

# Streamlit App
st.title("Speech-to-Text with Speaker Diarization")

# Upload audio file
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
    
    # Load and preprocess the audio file
    audio = AudioSegment.from_file(file_name)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    processed_audio_path = "processed_audio.wav"
    audio.export(processed_audio_path, format="wav")
    
    # Perform noise reduction
    st.write("Performing noise reduction...")
    audio_data, rate = librosa.load(processed_audio_path, sr=None)
    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate)
    noise_reduced_path = "noise_reduced_audio.wav"
    sf.write(noise_reduced_path, reduced_noise, rate)
    
    if os.path.exists(noise_reduced_path):
        st.write("Noise reduction completed successfully.")
    
    # Split audio into chunks for transcription
    def split_audio(audio_path, chunk_length_ms=30000):
        audio = AudioSegment.from_wav(audio_path)
        chunks = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunks.append(chunk)
        return chunks
    
    # Transcribe each chunk
    def transcribe_chunks(chunks):
        recognizer = sr.Recognizer()
        transcription = ""
        for i, chunk in enumerate(chunks):
            chunk.export(f"chunk{i}.wav", format="wav")
            audio_file = sr.AudioFile(f"chunk{i}.wav")
            with audio_file as source:
                audio_data = recognizer.record(source)
            try:
                chunk_transcription = recognizer.recognize_google(audio_data)
                transcription += chunk_transcription + " "
            except sr.UnknownValueError:
                st.write(f"Chunk {i}: Google Web Speech API could not understand the audio")
            except sr.RequestError as e:
                st.write(f"Chunk {i}: Could not request results from Google Web Speech API; {e}")
        return transcription
    
    # Split the noise-reduced audio
    chunks = split_audio(noise_reduced_path)
    
    # Transcribe the chunks
    transcription = transcribe_chunks(chunks)
    st.write("Transcription completed.")
    
    # Display transcription
    st.text_area("Transcription:", transcription)
    
    # Speaker Diarization
    st.write("Performing speaker diarization...")
    hf_token = "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"  # Your Hugging Face token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_token)
    
    if pipeline is not None:
        diarization = pipeline(noise_reduced_path)
        
        # Extract timestamps and speaker labels from diarization
        timestamps = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
        
        # Function to split transcription into segments based on timestamps
        def split_transcription(transcription, timestamps):
            words = transcription.split()
            n_words = len(words)
            segments = []
            for (start, end, speaker) in timestamps:
                n_segment_words = int((end - start) * n_words / (timestamps[-1][1] - timestamps[0][0]))
                segment = ' '.join(words[:n_segment_words])
                words = words[n_segment_words:]
                segments.append((segment, speaker))
            return segments
        
        # Split transcription into segments based on timestamps
        segments = split_transcription(transcription, timestamps)
        
        # Display the segmented transcription with speaker labels
        for segment, speaker in segments:
            st.write(f"Speaker {speaker}: {segment}")
    else:
        st.write("Error: Failed to load the speaker diarization pipeline.")

