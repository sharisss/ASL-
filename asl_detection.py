import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
from gtts import gTTS
import os

# Load the trained ASL model
model = tf.keras.models.load_model("asl_model.h5")  # Ensure you train your model before using

# Define ASL classes (modify based on dataset)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Del', 'Nothing']

# Initialize PyAudio for speech synthesis
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # Use 'afplay' on macOS or 'mpg321' on Linux
# Create folder if not exists
os.makedirs("captured_frames", exist_ok=True)
# Open Webcam for real-time ASL Detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (64, 64))  # Resize to match model input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize

    # Predict ASL sign
    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    asl_sign = classes[class_id]

    # Display prediction
    cv2.putText(frame, f"Detected: {asl_sign}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Detection", frame)

    # Convert sign to speech
    if cv2.waitKey(1) & 0xFF == ord('s'):
        text_to_speech(asl_sign)

    # Exit loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
