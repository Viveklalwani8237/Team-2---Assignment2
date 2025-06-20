from flask import Flask, request, jsonify
import joblib
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

app = Flask(__name__)

MODELS_DIR = "./model"
BEST_SPAM_MODEL_FILENAME = "best_spam_model_pipeline.pkl"
BEST_SPAM_MODEL_PATH = os.path.join(MODELS_DIR, BEST_SPAM_MODEL_FILENAME)
DATA_FILE = "spam_data.csv"

NLTK_STOPWORDS = set(stopwords.words('english'))
NLTK_LEMMATIZER = WordNetLemmatizer()

spam_model_pipeline = None
last_trained_params = {}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [NLTK_LEMMATIZER.lemmatize(word) for word in tokens if word not in NLTK_STOPWORDS]
    return ' '.join(tokens)

def load_model():
    global spam_model_pipeline
    if os.path.exists(BEST_SPAM_MODEL_PATH):
        try:
            spam_model_pipeline = joblib.load(BEST_SPAM_MODEL_PATH)
            print("Existing model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return False

load_model()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the Spam Classifier API",
        "available_endpoints": {
            "/train": "Train the spam classifier model (POST)",
            "/get_params": "Get last trained model hyperparameters (GET)",
            "/evaluate": "Predict if input text is spam or not (POST)"
        },
        "status": "running"
    }), 200

@app.route('/train', methods=['POST'])
def train_model():
    global spam_model_pipeline, last_trained_params

    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        return jsonify({"error": f"Dataset {DATA_FILE} not found. Please ensure it's in the same directory as the script."}), 500
    except Exception as e:
        return jsonify({"error": f"Error loading data: {str(e)}"}), 500

    required_cols = {'sms', 'label'}
    if not required_cols.issubset(data.columns):
        return jsonify({"error": f"CSV must contain columns: {required_cols}. Found: {list(data.columns)}"}), 500

    data['cleaned_text'] = data['sms'].apply(preprocess_text)
    X = data['cleaned_text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ])

    default_param_grid = {
        'lr__C': [1.0],
        'lr__penalty': ['l2'],
        'lr__solver': ['liblinear']
    }

    user_params = request.get_json(silent=True)
    param_grid_to_use = {}

    if user_params:
        valid_user_params = {}
        if "C" in user_params:
            valid_user_params['lr__C'] = [float(user_params["C"])]
        if "penalty" in user_params:
            valid_user_params['lr__penalty'] = [user_params["penalty"]]
        if "solver" in user_params:
            valid_user_params['lr__solver'] = [user_params["solver"]]

        param_grid_to_use.update(default_param_grid)
        param_grid_to_use.update(valid_user_params)
    else:
        param_grid_to_use = default_param_grid

    try:
        grid_search = GridSearchCV(pipeline, param_grid_to_use, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        optimal_hyperparameters = grid_search.best_params_
        last_trained_params = optimal_hyperparameters

        tuned_predictions = best_pipeline.predict(X_test)

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_pipeline, BEST_SPAM_MODEL_PATH)
        
        spam_model_pipeline = best_pipeline

        accuracy = accuracy_score(y_test, tuned_predictions)
        precision = precision_score(y_test, tuned_predictions)
        recall = recall_score(y_test, tuned_predictions)
        f1 = f1_score(y_test, tuned_predictions)
        conf_matrix = confusion_matrix(y_test, tuned_predictions).tolist()

        train_message = "Model training completed successfully." if not user_params else "Model re-trained with specified parameters."

        return jsonify({
            "message": train_message,
            "training_status": "completed successfully",
            "optimal_hyperparameters": optimal_hyperparameters,
            "evaluation_on_test_set": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix
            }
        }), 200

    except Exception as e:
        print(f"Error during training or evaluation: {str(e)}")
        return jsonify({"error": f"Error during model training or evaluation: {str(e)}"}), 500

@app.route('/get_params', methods=['GET'])
def get_trained_params():
    if not last_trained_params:
        return jsonify({"message": "No model has been trained yet, or parameters not recorded."}), 404
    return jsonify({"last_trained_parameters": last_trained_params}), 200

@app.route('/evaluate', methods=['POST'])
def evaluate_text():
    if spam_model_pipeline is None:
        return jsonify({"error": "Model not trained or loaded. Please train the model first."}), 400

    try:
        text_input = request.json['text']
    except KeyError:
        return jsonify({"error": "Please provide 'text' in the request body."}), 400

    preprocessed_text = preprocess_text(text_input)
    prediction = spam_model_pipeline.predict([preprocessed_text])[0]
    
    result = "spam" if prediction == 1 else "not spam"

    return jsonify({"input_text": text_input, "prediction": result}), 200

if __name__ == '__main__':
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    app.run(debug=True)
