from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained pipeline (TF-IDF + model + custom transformers, etc.)
model = joblib.load("msg_classifier_v1.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        # messages should be a string
        message = data["message"]

        # make a dataframe of a single column 'message' (required by the pipeline)
        message_df = pd.DataFrame({"message": [message]})

        # 0 -> ham , 1 -> spam

        # Run the pipeline
        pred_label = model.predict(message_df)[0]
        pred_proba = model.predict_proba(message_df)[0][1]  # Probability of the positive class (spam)


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
