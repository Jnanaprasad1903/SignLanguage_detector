import os
import pickle
import mediapipe as mp
import cv2


# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)


# Directory and data setup
DATA_DIR = './data'
data = []
labels = []


for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing class: {dir_}")

    for img_path in os.listdir(dir_path):
        full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(full_path)

        if img is None:
            print(f"Skipping unreadable image: {full_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)


        # Draw landmarks for visualization
        # Draw landmarks for visualization
        if results.multi_hand_landmarks:
            all_x = []
            all_y = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw detected hand landmarks on the image
                mp_drawing.draw_landmarks(
                    img,  # image to draw on
                    hand_landmarks,  # detected landmarks
                    mp_hands.HAND_CONNECTIONS,  # draw the connecting lines
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for lm in hand_landmarks.landmark:
                    all_x.append(lm.x)
                    all_y.append(lm.y)

            # Global normalization values
            min_x = min(all_x) if all_x else 0
            min_y = min(all_y) if all_y else 0
            
            left_hand = None
            right_hand = None
            
            for idx, hand_handedness in enumerate(results.multi_handedness):
                label = hand_handedness.classification[0].label # 'Left' or 'Right'
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
            
            # Append Left Hand (42 features)
            if left_hand:
                for lm in left_hand.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
            else:
                data_aux.extend([0.0] * 42)
                
            # Append Right Hand (42 features)
            if right_hand:
                for lm in right_hand.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
            else:
                data_aux.extend([0.0] * 42)

            data.append(data_aux)
            labels.append(dir_)

            # Show the image with landmarks
            cv2.imshow("Hand Landmarks", img)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

print(f"\nProcessed {len(data)} samples across {len(set(labels))} classes.")

# Save data to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data successfully saved to 'data.pickle'")

cv2.destroyAllWindows()
