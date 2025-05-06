Emotion Detection Web App – Setup Guide
This guide will walk you through the steps to clone, set up, train, and run the Emotion Detection application locally.

✅ Prerequisites
Python 3.7+

pip (Python package manager)

Git

Basic familiarity with Python and Flask

1. 🔁 Clone the Repository
Use Git to clone the repository to your local machine:

bash
Copy
Edit
git clone https://github.com/NAMan00018/Emotion_Detection.git
cd Emotion_Detection
2. 📦 Install Dependencies
Install the required Python libraries from the requirements.txt file:

bash
Copy
Edit
pip install -r requirements.txt
3. 🧾 Project Structure & File Overview
File/Folder	Description
train_model.py	Script to train the emotion classification model
app.py	Main Flask web application for emotion detection
predict_emotion.py	Helper script to process input and predict emotion
emotion_classifier.pkl	Pretrained model file (skip training if this exists)
emotion_dataset_raw.csv	Dataset used for training the model

4. 🧠 Train the Model (Optional)
You can skip this step if emotion_classifier.pkl is already provided.

To retrain the model:

bash
Copy
Edit
python train_model.py
This will train a new classifier and save it as emotion_classifier.pkl.

5. 🚀 Run the Web Application
Start the Flask application:

bash
Copy
Edit
python app.py
Once the server is running, open your browser and navigate to:

cpp
Copy
Edit
http://127.0.0.1:5000/
6. 🧪 Using the Application
Enter or upload text as instructed in the web interface.

The app will display the predicted emotion based on your input.

📸 Sample Interface
Below are example screenshots from the application interface:






🛠️ Troubleshooting Tips
Make sure your Python version is compatible with the required packages.

If any dependency fails, try upgrading pip or installing the packages individually.
