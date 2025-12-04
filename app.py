from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved ML model
saved = joblib.load('model.joblib')
model = saved['model']
target_names = saved['target_names']

@app.route('/')
def home():
    return "ML API is running. Use POST /predict."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'input' not in data:
            return jsonify({"error": "Send JSON with 'input': [values]"}), 400
        
        arr = np.array(data['input']).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr).tolist()[0]
        
        return jsonify({
            "prediction_index": int(pred),
            "prediction_name": target_names[int(pred)],
            "probabilities": proba
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
