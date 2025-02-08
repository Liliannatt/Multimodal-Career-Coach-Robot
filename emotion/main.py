import threading
import time
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import cv2
import numpy as np
from keras.losses import MeanSquaredError
import pyaudio
import wave
import azure.cognitiveservices.speech as speechsdk
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import threading
from keras_preprocessing.image import img_to_array

# Load Speech Emotion Recognition Model
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
audio_emotion_model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = audio_emotion_model.config.id2label

# Load Facial Emotion Recognition models
emotion_model = load_model('emotion_detection_model_50epochs.h5', custom_objects={'mse': MeanSquaredError()})
age_model = load_model('age_model_3epochs.h5', custom_objects={'mse': MeanSquaredError()})
gender_model = load_model('gender_model_3epochs.h5', custom_objects={'mse': MeanSquaredError()})
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

# Function for Facial Emotion Recognition
def live_face_detection(duration=200, output_emotion_file='output_facial_emotion.txt', 
                        output_gender_file='output_gender.txt', 
                        output_age_file='output_age.txt',
                        output_emotion_ratio_file='output_facial_emotion_ratio.txt'):
    emotion_model = load_model('emotion_detection_model_50epochs.h5', custom_objects={'mse': MeanSquaredError()})
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    gender_labels = ['Male', 'Female']

    # Variables to store data for averaging
    emotion_counts = {label: 0 for label in class_labels}
    age_sum = 0
    gender_counts = {label: 0 for label in gender_labels}
    frame_count = 0  # Total number of frames with detected faces

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        labels = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            frame_count += 1  # Increment frame count for each detected face

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Emotion Prediction
            roi = roi_gray.astype('float') / 255.0  # Scaling the image
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

            preds = emotion_model.predict(roi)[0]  # One hot encoded result for 7 classes
            confidence = preds.max() * 100  # Get the confidence in percentage
            label = f"{class_labels[preds.argmax()]} ({confidence:.2f}%)"  # Add confidence to the label
            emotion_counts[class_labels[preds.argmax()]] += 1  # Increment count for this emotion
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Gender Prediction
            roi_color = frame[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (200, 200), interpolation=cv2.INTER_AREA)
            gender_predict = gender_model.predict(np.array(roi_color).reshape(-1, 200, 200, 3))
            gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
            gender_label = gender_labels[gender_predict[0]]
            gender_counts[gender_label] += 1  # Increment count for this gender
            gender_label_position = (x, y + h + 50)  # 50 pixels below to move the label outside the face
            cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Age Prediction
            age_predict = age_model.predict(np.array(roi_color).reshape(-1, 200, 200, 3))
            age = round(age_predict[0, 0])
            age_sum += age  # Add age to the total
            age_label_position = (x + h, y + h)
            cv2.putText(frame, "Age=" + str(age), age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Face Detection', frame)

        # Break the loop after the specified duration or if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > duration:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video feed terminated after 200 seconds.")

    # Calculate and write the average statistics to files
    if frame_count > 0:
        # Compute average emotion, gender, and age
        average_emotion = max(emotion_counts, key=emotion_counts.get)
        average_gender = max(gender_counts, key=gender_counts.get)
        average_age = age_sum / frame_count

        # Compute emotion ratios
        emotion_ratios = {label: (count / frame_count) * 100 for label, count in emotion_counts.items()}

        # Write results to text files
        with open(output_emotion_file, 'w') as f:
            f.write(average_emotion)
        with open(output_gender_file, 'w') as f:
            f.write(average_gender)
        with open(output_age_file, 'w') as f:
            f.write(f"{average_age:.2f}")
        with open(output_emotion_ratio_file, 'w') as f:
            for emotion, ratio in emotion_ratios.items():
                f.write(f"{emotion}: {ratio:.2f}%\n")

        print(f"Gender: {average_gender}")
        print(f"Age: {average_age}")
        print(f"Facial Emotion: {average_emotion}")
        print("Facial Emotion Ratios:")
        for emotion, ratio in emotion_ratios.items():
            print(f"{emotion}: {ratio:.2f}%")
    else:
        print("No faces detected during the session.")

# Audio Recording Function
def record_audio(file_name="output.wav", duration=200, chunk_size=1024, format_type=pyaudio.paInt16, channels=2, sample_rate=44100):
    p = pyaudio.PyAudio()

    stream = p.open(format=format_type,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("* recording")

    frames = []

    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format_type))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function for preprocessing audio
def preprocess_audio(audio_path, feature_extractor, max_duration=200.0):
    """
    Preprocess audio for Whisper model with support for 200 seconds input.
    Ensures mel features are correctly padded or truncated.
    """
    # Load the audio file
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)

    # Calculate the maximum number of samples for 200 seconds
    max_length = int(feature_extractor.sampling_rate * max_duration)
    
    # Truncate or pad the audio array to fit the 200 seconds duration
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]  # Truncate to 200 seconds
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)), mode='constant')

    # Extract features compatible with the Whisper model
    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        truncation=True,
    )
    
    return inputs

def chunk_audio_features(inputs, chunk_size=3000):
    """
    Split mel features into chunks of size `chunk_size`.
    """
    mel_features = inputs['input_features']
    num_chunks = mel_features.size(-1) // chunk_size
    
    # Ensure proper chunking
    mel_chunks = torch.split(mel_features, chunk_size, dim=-1)
    return mel_chunks

# Audio Emotion Detection
def predict_audio_emotion(audio_path, model, feature_extractor, id2label, max_duration=200.0, output_file='output_audio_emotion.txt'):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    
    # Chunk the features
    mel_chunks = chunk_audio_features(inputs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predicted_labels = []
    for chunk in mel_chunks:
        chunk = chunk.to(device)
        with torch.no_grad():
            outputs = model(input_features=chunk)
        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_labels.append(id2label[predicted_id])

    # Write predictions to a file
    with open(output_file, 'w') as txt_file:
        for prediction in predicted_labels:
            txt_file.write(f"{prediction}\n")

    return predicted_labels

# Function for Speech to Text
def speech_to_text(file_name='output.wav', subscription_key="0e2afde1930544d297072fd3fe1e5f40", region="swedencentral", output_file='output_text.txt'):
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)

    audio_config = speechsdk.audio.AudioConfig(filename=file_name)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))

        # Write result to a text file
        with open(output_file, 'w') as txt_file:
            txt_file.write(result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

    return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else None

# Threaded Execution
def main():
    # Start facial detection in a separate thread
    face_thread = threading.Thread(
        target=live_face_detection, 
        kwargs={
            'duration': 200, 
            'output_emotion_file': 'output_facial_emotion.txt', 
            'output_gender_file': 'output_gender.txt', 
            'output_age_file': 'output_age.txt',
            'output_emotion_ratio_file': 'output_facial_emotion_ratio.txt'
        }
    )
    face_thread.start()

    # Start audio recording in the main thread (runs concurrently with face detection)
    audio_file = "output.wav"
    record_audio(file_name=audio_file, duration=200)

    # Perform audio emotion detection after recording is complete
    emotion = predict_audio_emotion(
        audio_file, 
        audio_emotion_model, 
        feature_extractor, 
        id2label, 
        output_file='output_audio_emotion.txt'
    )
    print(f"Audio Emotion: {emotion}")

    # Wait for the face detection thread to finish
    face_thread.join()

    # Perform speech-to-text
    transcription = speech_to_text(file_name=audio_file, output_file='output_text.txt')
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    main()