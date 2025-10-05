from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Setup Flask and point templates/static to frontend
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
template_folder = os.path.join(base_dir, 'frontend', 'templates')
static_folder = os.path.join(base_dir, 'frontend', 'static')

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

# Cargar el modelo una sola vez al iniciar el servidor
try:
    model = joblib.load("model.joblib")
except Exception:
    model = None


@app.route('/mapa')
def mapa():
    """Sirve la página con el visor 3D (mapa de exoplanetas)."""
    return render_template('mapa.html')


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500
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
