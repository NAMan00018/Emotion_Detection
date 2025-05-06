import pandas as pd
import neattext.functions as nfx
from sklearn.model_selection import train_test_split, GridSearchCV  # <-- Import GridSearchCV here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline  # <-- Import Pipeline
from nltk.stem import WordNetLemmatizer
import joblib
import nltk

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("emotion_dataset_raw.csv")

# Rename columns if needed
if 'Text' in df.columns and 'Emotion' in df.columns:
    df.rename(columns={'Text': 'text', 'Emotion': 'emotion'}, inplace=True)
elif 'text' not in df.columns or 'emotion' not in df.columns:
    raise ValueError("❌ Required columns 'text' and 'emotion' not found.")

# Clean text: Remove stopwords, special characters, and apply lemmatization
lemmatizer = WordNetLemmatizer()
df['clean_text'] = df['text'].apply(nfx.remove_stopwords).apply(nfx.remove_special_characters)
df['clean_text'] = df['clean_text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

# Split data into training and testing sets
X = df['clean_text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline with TfidfVectorizer and Logistic Regression
model = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Hyperparameter tuning for Logistic Regression using GridSearchCV
param_grid = {'clf__C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Train the model
model = grid_search.best_estimator_

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📊 Accuracy on test set: {accuracy:.2f}")
print("📄 Classification Report:")
print(report)

# Save model
joblib.dump(model, "emotion_classifier.pkl")
print("💾 Model saved as emotion_classifier.pkl")
