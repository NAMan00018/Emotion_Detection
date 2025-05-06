import joblib
import neattext.functions as nfx

# Load the trained model
model = joblib.load("emotion_classifier.pkl")

def predict_emotion(text):
    clean_text = nfx.remove_stopwords(nfx.remove_special_characters(text))
    prediction = model.predict([clean_text])
    return prediction[0]

if __name__ == "__main__":
    user_input = input("Enter a sentence: ")
    emotion = predict_emotion(user_input)
    print("Predicted Emotion:", emotion)
