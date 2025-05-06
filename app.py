from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import neattext.functions as nfx

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the dataset
def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv("emotion_dataset_raw.csv")

    # Clean the text
    df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

    # Split the dataset into input features and target labels
    x = df['Clean_Text']
    y = df['Emotion']

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test

# Train the model
def train_model(x_train, y_train):
    # Create a pipeline with a CountVectorizer and Logistic Regression
    pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])
    pipe_lr.fit(x_train, y_train)
    
    # Save the trained model
    joblib.dump(pipe_lr, "emotion_classifier.pkl")
    return pipe_lr

# Define the emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", 
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Prediction function
def predict_emotions(docx, model):
    results = model.predict([docx])
    return results[0]

# Prediction probability function
def get_prediction_proba(docx, model):
    results = model.predict_proba([docx])
    return results

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    emoji_icon = None
    confidence = None

    # Load and preprocess data
    x_train, x_test, y_train, y_test = load_and_preprocess_data()

    # Train the model (only once when the page loads or at first use)
    model = train_model(x_train, y_train)

    if request.method == "POST":
        # Get text from form
        raw_text = request.form["raw_text"]

        # Get the prediction and probability
        prediction = predict_emotions(raw_text, model)
        probability = get_prediction_proba(raw_text, model)
        emoji_icon = emotions_emoji_dict[prediction]
        confidence = np.max(probability)

        return render_template("index.html", 
                               raw_text=raw_text, 
                               prediction=prediction,
                               emoji_icon=emoji_icon, 
                               confidence=confidence)

    return render_template("index.html", prediction=prediction)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

