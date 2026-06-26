import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

# Load trained model
# Ensure 'model.p' is in the correct location or provide a full path
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Webcam setup
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Increased confidence for potentially better accuracy
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

# Labels mapping (1–9, A–Z)
labels_dict = {
    1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
    18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P',
    26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
    34: 'Y', 35: 'Z'
}

# Speech engine
engine = pyttsx3.init()
# Slow down the speech rate (default is usually around 200)
engine.setProperty('rate', 140)



# Detection timing
last_prediction_time = 0
# Reduced interval to 3 seconds for a smoother experience
prediction_interval = 3

# Store detected string
detected_string = ""
# Store the character being currently displayed (even before it's added to the string)
current_display_char = ""
current_display_confidence = None

print("\nPress 'q' → Quit\n")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for a natural mirror-like view
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
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
        
        # Predict every 'prediction_interval' seconds
        if current_time - last_prediction_time > prediction_interval:
            
            # Predict
            try:
                prediction_value = model.predict([np.asarray(data_aux)])[0]
                
                # Get confidence (if model supports predict_proba)
                proba = model.predict_proba([np.asarray(data_aux)])
                confidence = max(proba[0])
            except:
                prediction_value = None
                confidence = None

            # Map prediction value (e.g., 10 for 'A') to the label string ('A')
            # Assuming prediction_value is an integer key from the trained model
            # Your existing code suggests prediction[0] is already a string, but this is safer
            predicted_char = str(prediction_value)
            
            # Update display character and confidence immediately
            current_display_char = predicted_char
            current_display_confidence = confidence

            # Append character to string ONLY if a successful prediction was made
            if predicted_char:
                # Prevent spamming the same word consecutively while holding the sign
                words = detected_string.split()
                if not words or words[-1] != predicted_char:
                    if detected_string:
                        detected_string += " " + predicted_char
                    else:
                        detected_string += predicted_char

                last_prediction_time = current_time

                # Print the accumulating string to the console
                print(f"Detected string: **{detected_string}** (Added: {predicted_char})")
                
        else:
            # While waiting for the interval, just keep the last successful prediction on screen
            pass
            
    # If no hands are detected, clear the current display character
    else:
        current_display_char = ""
        current_display_confidence = None


    # --- Display Current State in Video Frame ---
    
    # Text to show the currently accumulating string
    cv2.putText(frame, f"Sequence: {detected_string}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Text to show the current prediction (or the last one if in the interval)
    display_text = current_display_char
    if current_display_confidence is not None:
        display_text += f" ({current_display_confidence*100:.1f}%)"
        
    cv2.putText(frame, display_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('ASL Detector | Press "q" to Quit', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- Cleanup and Final Output ---
cap.release()
cv2.destroyAllWindows()

# Final output as requested
print("\n--- Detection Ended ---")
print("Final detected string:", detected_string)

if detected_string:
    # Speak the final string
    engine.say(detected_string)
    engine.runAndWait()
else:
    print("No characters were detected.")
    
engine.stop()