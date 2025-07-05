from flask import Flask, request, jsonify , render_template
import pickle
import re
import pandas as pd

app = Flask(__name__)

with open("Spam_detection_RG.pkl","rb") as fileobj:
        model = pickle.load(fileobj)

def predict_message(text):
    features = extract_features(text)
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    return "Seems to be Spam" if prediction == 1 else "No worry, this is not spam"


def extract_features(email):
    features = {}

    # Length of email
    features['email_len'] = len(email)

    # Number of uppercase words
    features['num_uppercase_words'] = len(re.findall(r'\b[A-Z]{2,}\b', email))

    # Number of exclamation marks
    features['num_exclamations'] = email.count('!')

    # Number of URLs
    features['num_links'] = len(re.findall(r'http[s]?://', email))

    # Presence of HTML
    features['has_html'] = int(bool(re.search(r'<[^>]+>', email)))

    # Presence of spammy_words
    spammy_words = [
    'hurry', 'limited', 'win', 'few', 'credited', 'cash prize', 'click',
    'congratulations', 'lottery', 'free', 'urgent', 'act now', 'exclusive',
    'guaranteed', 'winner', 'miracle', 'earn money', 'get paid', 'easy money',
    'no cost', 'risk-free', 'special promotion', 'buy now', 'order now',
    'instant access', 'double your income', 'extra cash', 'financial freedom',
    'lowest price', 'money back', 'offer expires', 'trial', 'unsecured credit',
    'weight loss', 'viagra', 'investment', 'billion', 'million dollars', 'high return','earn more', 'earn money from home','Free Entry']

    features['Word_presence'] = int(any(word.lower() in email.lower() for word in spammy_words))

    if features['Word_presence'] >=1:
             features['Word_presence'] =   features['Word_presence']*5
    return features


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/RG_spam_Detection", methods=['POST'])
def predict():
    Text = request.form.get("message")
    Classification = predict_message(Text)
    return render_template("index.html", result= Classification)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
