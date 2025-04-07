from flask import Flask, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

# Load the model
model_dict = pickle.load(open('./model.pkl', 'rb'))
model = model_dict['model']

# Set up Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels
labels_dict = {
    0: 'Hello',
    1: 'okay',
    2: 'My name is',
    3: 'where',
    4: 'eat',
    5: 'drink',
    6: 'pain',
    7: 'yes',
    8: 'no',
    9: 'thank you'
}

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Load and convert image
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    H, W, _ = frame.shape

    # Run Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({'error': 'No hand detected'}), 400

    hand_landmarks = results.multi_hand_landmarks[0]

    x_ = []
    y_ = []
    data_aux = []

    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)

    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min(x_))
        data_aux.append(lm.y - min(y_))

    data_aux = np.asarray(data_aux).reshape(1, -1)
    prediction = model.predict(data_aux)
    predicted_idx = int(prediction[0])
    predicted_label = labels_dict.get(predicted_idx, f"Unknown {predicted_idx}")

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
