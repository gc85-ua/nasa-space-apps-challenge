from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo una sola vez al iniciar el servidor
model = joblib.load("model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Esperamos JSON con una lista de características
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)  # Ajusta reshape según tu modelo
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
