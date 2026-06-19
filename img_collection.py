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
print("Type 'exit' to quit.\n")

while True:

    class_name = input("Enter class label (or type 'exit' to stop): ").strip()
    if class_name.lower() == 'exit':
        break

    
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    print(f"Ready to collect data for class: '{class_name}'")
    print("Video is ON. Type 's' and press Enter to start capturing images...\n")

    
    start = False
    user_input_ready = False
    user_input_value = None
    
    import threading
    
    def read_input():
        global user_input_ready, user_input_value
        user_input_value = input().strip().lower()
        user_input_ready = True
    
    input_thread = threading.Thread(target=read_input, daemon=True)
    input_thread.start()
    
    while not start:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {class_name} | Press 's' to start",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        try:
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)
        except:
            pass
        
        if user_input_ready:
            if user_input_value == 's':
                start = True
            elif user_input_value == 'q' or user_input_value == 'exit':
                cap.release()
                cv2.destroyAllWindows()
                exit()
            user_input_ready = False
            input_thread = threading.Thread(target=read_input, daemon=True)
            input_thread.start()


    # Start capturing images
    counter = 0
    print("Capturing images... Type 'q' and press Enter to stop for this class.")
    
    import threading
    stop_capture = False
    
    def wait_for_stop():
        global stop_capture
        while not stop_capture:
            user_input = input().strip().lower()
            if user_input == 'q':
                stop_capture = True
                break
    
    stop_thread = threading.Thread(target=wait_for_stop, daemon=True)
    stop_thread.start()

    while not stop_capture:
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
        try:
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(25)
        except:
            pass
        
        if counter >= 100:  # Collect 100 images per class automatically
            print(f"Collected 100 images for class '{class_name}'")
            stop_capture = True

    print(f"Stopped capturing '{class_name}'. Total: {counter} images.\n")
    stop_capture = False

print("\nData collection finished.")
cap.release()
cv2.destroyAllWindows()
