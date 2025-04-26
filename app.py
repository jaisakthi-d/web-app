from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)  # No template_folder needed since it’s the default 'templates'

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Error rendering template: {e}", 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form and convert to float
        features = [float(x) for x in request.form.values()]
        data = np.array([features])
        prediction = model.predict(data)

        result = "Positive" if prediction[0] == 1 else "Negative"
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    except Exception as e:
        return f"Error during prediction: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)