# main.py — FastAPI for RandomForest Spam Detector + Feedback

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import re
import pickle
import os
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# File paths
MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
PARAMS_PATH = "best_params.txt"
DATA_PATH = "train.csv"
FEEDBACK_PATH = "feedback_data.csv"

app = FastAPI()

class EmailInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    true_label: int  # 0 for ham, 1 for spam

# Load model and vectorizer if exist
model = pickle.load(open(MODEL_PATH, "rb")) if os.path.exists(MODEL_PATH) else None
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb")) if os.path.exists(VECTORIZER_PATH) else None

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", " url ", text)
    text = re.sub(r"[^a-zA-Z0-9!?\\s]", "", text)
    return text

@app.get("/best_model_parameter")
def best_model_param():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            return {"best_model_parameters": f.read()}
    return {"error": "Model not trained yet."}

@app.post("/predict")
def predict_spam(email: EmailInput):
    global model, vectorizer
    if not model or not vectorizer:
        return {"error": "Model not trained yet."}
    cleaned = clean_text(email.text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    return {"prediction": "spam" if pred == 1 else "ham"}

@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput):
    cleaned = clean_text(feedback.text)
    if not os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "w") as f:
            f.write("sms,label\n")
    with open(FEEDBACK_PATH, "a") as f:
        f.write(f'"{cleaned}",{feedback.true_label}\n')
    return {"message": "Feedback recorded successfully ✅"}

@app.post("/train")
def train_model():
    global model, vectorizer

    # Load original data
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    df["sms"] = df["sms"].apply(clean_text)

    # Merge feedback if exists
    if os.path.exists(FEEDBACK_PATH):
        df_fb = pd.read_csv(FEEDBACK_PATH)
        df_fb.dropna(inplace=True)
        df_fb["sms"] = df_fb["sms"].apply(clean_text)
        df = pd.concat([df, df_fb], ignore_index=True)

    X_text, y = df["sms"], df["label"]
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X_text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(RandomForestClassifier(class_weight="balanced"), param_grid, cv=3, scoring="f1")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(best_model, "rf_model")

        # Save model and vectorizer
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_model, f)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(PARAMS_PATH, "w") as f:
            f.write("RandomForestClassifier with GridSearchCV\n")
            for k, v in grid_search.best_params_.items():
                f.write(f"{k}={v}\n")
            f.write(f"accuracy={acc}\nf1_score={f1}\nroc_auc={auc}\n")

        model = best_model

    return {"message": "Model trained successfully", "accuracy": acc, "f1_score": f1, "roc_auc": auc}
