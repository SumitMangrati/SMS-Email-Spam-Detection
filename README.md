# SMS Spam Detection Chrome Extension

This project is a **Chrome Extension** that detects SMS spam using a **Machine Learning Model** implemented with Python and Flask for the backend. The model is trained on a dataset of SMS messages and classifies them as **spam** or **ham (not spam)**. The extension allows users to check messages in real time for potential spam content.

## API Link 
https://sms-spam-detector-zhys.onrender.com/predict
(Deployed via Render)

## Screenshots
![Screenshot 2025-03-26 221903](https://github.com/user-attachments/assets/114a1a94-2124-4158-8029-ddea2de9b7e3)

## Workflow Diagram

![diagram-export-3-26-2025-10_35_36-PM](https://github.com/user-attachments/assets/e20e00b7-ac80-4224-b777-f8f1896f5e0a)

## Installation and Setup

### 1. Clone the Repository
```
git clone https://github.com/SumitMangrati/SMS-Email-Spam-Detection.git
```
### 2. Navigate to the project directory
```
cd SMS-Email-Spam-Detection
```
### 3. Create a new environment
```
conda create -p venv python==3.7 -y
```
### 4. Activate the environment
```
conda activate venv/
```
### 5. Install the requirements
```
pip install -r backend/requirements.txt
```
### 6. Running API locally
```
python backend/app.py
```
API runs on `http://127.0.0.1:5000/predict`

### 7. Running API via docker
```
docker build -t sms-spam-detector .
docker run -p 5000:5000 sms-spam-detector
```
API runs on `http://localhost:5000/predict`

## Setting Up the Chrome Extension
1. Open **Google Chrome** and go to `chrome://extensions/`.
2. Enable **Developer Mode** (top-right corner).
3. Click **Load unpacked** and select the `extension/` folder.
4. The extension will be added to your browser.

## Features
- Detects spam messages with high accuracy.
- Uses a Machine Learning model trained on SMS data.
- Backend built with Flask for model inference.
- Simple and user-friendly Chrome extension interface.


## Machine Learning Model
The model is trained on the **SMS Spam Collection Dataset**, which contains labeled SMS messages as spam or ham. The steps include:

1. **Data Preprocessing:**
   - Removing special characters and stopwords.
   - Converting text to lowercase.
   - Tokenization and vectorization using **TF-IDF**.

2. **Model Training:**
   - Trained a **Naive Bayes classifier** for text classification.
   - Achieved high accuracy in detecting spam.

3. **Deployment:**
   - The trained model is saved as `model.pkl`.
   - The TF-IDF vectorizer is saved as  `vectorizer.pkl`.
   - Flask backend loads the model and provides an API for prediction.
  
## Contribution
If you’d like to contribute, feel free to submit a pull request or open an issue

