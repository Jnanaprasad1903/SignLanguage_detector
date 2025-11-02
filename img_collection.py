import os
import cv2


DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


cap = cv2.VideoCapture(0)

# ROI coordinates (Region of Interest)
x1, y1 = 150, 100
x2, y2 = 450, 400

print("Webcam is ON")
print("Press 'q' anytime to quit.\n")

while True:

    class_name = input("Enter class label (or type 'exit' to stop): ").strip()
    if class_name.lower() == 'exit':
        break

    
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    print(f"Ready to collect data for class: '{class_name}'")
    print("Press 's' to start capturing images...")

    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {class_name} | Press 's' to start",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()


    # Start capturing images
    counter = 0
    print("Capturing images... Press 'q' to stop for this class.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Extract region of interest
        roi = frame[y1:y2, x1:x2]

        # Save frame
        save_path = os.path.join(class_dir, f"{counter}.jpg")
        cv2.imwrite(save_path, roi)
        counter += 1

        # Display frame with live counter
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {class_name} | Captured: {counter}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'): 
            print(f"Stopped capturing '{class_name}'. Total: {counter} images.\n")
            break

print("\nData collection finished.")
cap.release()
cv2.destroyAllWindows()
