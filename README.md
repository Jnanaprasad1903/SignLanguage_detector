# Sign Language Detector

This project is a real-time American Sign Language (ASL) detector. It uses a custom dataset collected from your webcam, extracts hand landmarks using **MediaPipe**, and classifies the signs using a **Random Forest Classifier** from scikit-learn. The final output uses text-to-speech to speak the detected string.

## Project Structure
- `img_collection.py`: Script to collect training images from your webcam.
- `dataset_creation.py`: Script to process the images, extract MediaPipe landmarks, and save them to a dataset (`data.pickle`).
- `train_classifier.py`: Script to train the Random Forest model on your dataset and save it (`model.p`).
- `test.py`: The live application that uses the trained model to detect signs in real-time and speak the words.

## Prerequisites
I have already fixed your `requirements.txt` (removed the incompatible `pickle5` library, since Python 3.8+ handles this natively) and installed all necessary dependencies in your virtual environment!

## How to Run the Project (Step-by-Step)
Since the project relies heavily on your webcam and interactive keyboard input, **please run these commands sequentially in your own terminal**.

### Step 1: Collect Data
```bash
python img_collection.py
```
- The console will ask for a class label (e.g., `A`, `B`, `C` or `1`, `2`, `3`).
- Put your hand in the green square on the webcam.
- Press `s` in your terminal and press Enter to start capturing 100 images for that class.
- Repeat for as many signs as you want to train. Type `exit` when done.

### Step 2: Extract Features
```bash
python dataset_creation.py
```
- This will process the images saved in the `./data` folder.
- It extracts the 21 3D hand landmarks using MediaPipe and saves them into a `data.pickle` file.

### Step 3: Train the Model
```bash
python train_classifier.py
```
- This script loads the `data.pickle` file.
- It trains a Random Forest Machine Learning model.
- It will print out the accuracy score and save the trained model as `model.p`.

### Step 4: Run Real-time Detection
```bash
python test.py
```
- This opens your webcam for real-time detection.
- Make the signs you trained inside the green box.
- The script will accumulate characters into a string and display them on the screen.
- Press `q` to quit. The application will then speak the accumulated sentence out loud!
