# Advanced Sign Language Detector

This project is a real-time, **Double-Handed** American Sign Language (ASL) detector. Built with Flask, MediaPipe, OpenCV, and a Random Forest classifier, it tracks hand landmarks, normalizes spatial distances, and predicts whole words. It features a sleek, modern web UI that translates signs into spoken English sentences using native browser Text-to-Speech (TTS).

## ✨ Key Features & Upgrades

- **Double Hand Support (84-Feature Array):** Unlike basic implementations, this AI model strictly expects 84 features (42 for the Left Hand, 42 for the Right Hand). If one hand is missing, it pads the data with zeros. This allows it to distinguish between single-handed and double-handed signs seamlessly.
- **Global Normalization:** Hand coordinates are normalized across a global bounding box spanning *both* hands. This preserves the physical distance between hands (crucial for ASL) while making the model immune to how close or far you stand from the camera.
- **Word-Level Detection:** Instead of spelling letter-by-letter, this model is designed to detect whole words. It filters out consecutive duplicate frames to build smooth, readable sentences.
- **Premium Web UI:** A beautiful dark-mode glassmorphic web app that streams your webcam, displays the live prediction, and accumulates translated sentences.
- **Native Browser TTS:** Text-to-Speech is offloaded from Python to the browser's native `SpeechSynthesis` API, ensuring unblocked, high-quality audio feedback.

## 📁 Project Structure

- `app.py`: Flask backend that serves the UI, runs MediaPipe, and streams the MJPEG video feed.
- `templates/index.html`: The HTML structure for the web interface.
- `static/style.css`: Premium Vanilla CSS styling with glassmorphism effects.
- `static/script.js`: Handles real-time DOM updates and Web Speech API TTS.
- `img_collection.py`: Utility to capture raw webcam frames into the `data/` folder. (Scans the full camera frame, no restrictive ROI box!)
- `dataset_creation.py`: Extracts and normalizes the 84-feature hand landmarks, saving them to `data.pickle`.
- `train_classifier.py`: Trains the Random Forest classifier and outputs `model.p`.

## 🚀 Setup & Installation

1. Make sure you have Python 3.8+ installed.
2. Install the required dependencies inside your virtual environment:
```bash
pip install -r requirements.txt
```

## 🧠 Training Your Own Signs

Because this model uses an advanced 84-feature input for double-hand support, you must train your own custom signs before using the app!

### 1. Collect Images
Run the data collector. Type the name of the word you want to train (e.g., "Hi", "Done", "Help"), then press `s` to start capturing 100+ images. Press `q` to stop.
```bash
python img_collection.py
```

### 2. Build the Dataset
Extract the 84-feature hand landmarks from your collected images. This mathematically maps left and right hands to their correct slots and applies global normalization.
```bash
python dataset_creation.py
```

### 3. Train the Classifier
Train the Random Forest model on your new data. The script will automatically evaluate the model on a test split, print a detailed **Classification Report (Accuracy, Precision, Recall)**, and save it to `model.p`.
```bash
python train_classifier.py
```

## 🌐 Running the Web App

Once your model is trained, start the Flask web server:

```bash
python app.py
```

Open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

*Enjoy your custom ASL Translator!*
