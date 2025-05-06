Clone the Repository
Use Git to clone the repository to your local machine:
bash
git clone https://github.com/NAMan00018/Emotion_Detection.git
Navigate into the directory:
bash
cd Emotion_Detection
2. Install Dependencies
Install required Python libraries listed in requirements.txt:
bash
pip install -r requirements.txt
3. Understand the Files
train_model.py: Script to train the emotion classifier.
app.py: The main web app for emotion detection.
predict_emotion.py: Script to predict emotions, possibly used by the app.
emotion_classifier.pkl: The already trained model (you can skip re-training unless needed).
emotion_dataset_raw.csv: Dataset used for training.
4. Train the Model (if you want to retrain)
Make sure your environment has the necessary libraries.
Run the training script:
bash
python train_model.py
This will train a new model and save it as emotion_classifier.pkl.
5. Run the App
Start the Flask application:
bash
python app.py
Open your browser and go to http://127.0.0.1:5000/ to access the emotion detection interface.
6. Use the App
Upload text or input data as instructed in the app interface.
The app will predict the emotion based on the input.


![image](https://github.com/user-attachments/assets/e93ef12d-2105-4f94-a93d-e4c06f4753b9)
![image](https://github.com/user-attachments/assets/54746e5f-8aae-4623-b5f3-2c3cfebf9865)
![image](https://github.com/user-attachments/assets/40255312-4c86-4ace-9553-928d9c8a8322)
![image](https://github.com/user-attachments/assets/cde7e511-1c35-4151-b8d1-8543402e93b4)

