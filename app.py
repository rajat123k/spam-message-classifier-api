from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.stem import PorterStemmer   # port stemmer
import re
import string
import pandas as pd

# download nltk resources
nltk.download('stopwords')
nltk.download('punkt_tab')

app = Flask(__name__)

# create a function to preprocess text
# get stop words
with open('stopwords.txt', 'r') as f:
  stop_words = f.read()

stopwords = stop_words.split()

def preprocess_text(text):
    # convert text to lower case
    text = text.lower()

    # remove punctuation and stopwords
    patt = f'[{string.punctuation}]' + '|' + '\\b(' + f'{'|'.join(stopwords)}' + ')\\b'
    text = re.sub(patt, ' ', text)

    # tokenize
    text = nltk.word_tokenize(text)

    # stemming
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]

    # convert text(list) into string
    return ' '.join(text)


# Load your trained pipeline (Tfidf --> model)
model = joblib.load("msg_classifier_v1.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing Data"}), 400

        # String data is passed in the 'message' key
        message = data["message"]
        message = pd.Series(message)  # convert to pandas series
        # preprocess the message
        message = message.apply(preprocess_text) # now message is a preprocessed string
        print(message)

        # 0 = ham , 1 = spam

        # Run the pipeline
        pred_label = model.predict(message)
        pred_proba = model.predict_proba(message)[0, 1]  # Probability of the positive class (spam)
        print(pred_proba)

        # Assuming classes are ['ham', 'spam']

        return jsonify({
            "prediction": "spam" if pred_label == 1 else "Not Spam",
            "spam_probability": pred_proba,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # development mode
    app.run(host="0.0.0.0", port=5000, debug=True)


'''
numpy 2.0.2
pandas 2.2.2
nltk 3.9.1
sklearn 1.6.1
joblib 1.5.2
re 2.2.1
'''