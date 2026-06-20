import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Load trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading model.p. Make sure you have trained the model first.")
    model = None

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Global State Variables
detected_string = ""
current_display_char = ""
current_display_confidence = None
last_prediction_time = 0
prediction_interval = 2.5 # Time between word registrations
new_word_detected = False # To trigger TTS on the frontend

def generate_frames():
    global detected_string, current_display_char, current_display_confidence, last_prediction_time, new_word_detected
    
    cap = cv2.VideoCapture(0)
    


    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks and model is not None:
            all_x = []
            all_y = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for lm in hand_landmarks.landmark:
                    all_x.append(lm.x)
                    all_y.append(lm.y)
                    
            min_x = min(all_x) if all_x else 0
            min_y = min(all_y) if all_y else 0
            
            left_hand = None
            right_hand = None
            
            for idx, hand_handedness in enumerate(results.multi_handedness):
                label = hand_handedness.classification[0].label
                if label == "Left" and left_hand is None:
                    left_hand = results.multi_hand_landmarks[idx]
                elif label == "Right" and right_hand is None:
                    right_hand = results.multi_hand_landmarks[idx]
                else:
                    if left_hand is None:
                        left_hand = results.multi_hand_landmarks[idx]
                    else:
                        right_hand = results.multi_hand_landmarks[idx]
                        
            data_aux = []
            
            if left_hand:
                for lm in left_hand.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
            else:
                data_aux.extend([0.0] * 42)
                
            if right_hand:
                for lm in right_hand.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
            else:
                data_aux.extend([0.0] * 42)
                    
            current_time = time.time()
            
            # Predict
            try:
                prediction_value = model.predict([np.asarray(data_aux)])[0]
                proba = model.predict_proba([np.asarray(data_aux)])
                confidence = max(proba[0])
                predicted_char = str(prediction_value)
                
                current_display_char = predicted_char
                current_display_confidence = confidence
                
                # Append if interval passed
                if current_time - last_prediction_time > prediction_interval:
                    words = detected_string.split()
                    if not words or words[-1] != predicted_char:
                        if detected_string:
                            detected_string += " " + predicted_char
                        else:
                            detected_string += predicted_char
                            
                        new_word_detected = True # Flag for frontend to speak
                        
                    last_prediction_time = current_time
            except Exception as e:
                pass
                
        else:
            current_display_char = ""
            current_display_confidence = None

        # Encode frame to stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_state')
def get_state():
    global new_word_detected
    
    response = {
        'sentence': detected_string,
        'current_word': current_display_char,
        'confidence': current_display_confidence,
        'trigger_speak': new_word_detected
    }
    
    if new_word_detected:
        new_word_detected = False # Reset after sending
        
    return jsonify(response)

@app.route('/clear')
def clear():
    global detected_string
    detected_string = ""
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
