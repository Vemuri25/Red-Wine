from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import pandas as pd
import os

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = os.environ.get("MODEL_PATH", "winequality-red.onnx")

# Load ONNX model
try:
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
except Exception as e:
    sess = None
    input_name = None
    print("Failed to load ONNX model:", e)

@app.route('/predict', methods=['POST'])
def predict():
    if sess is None:
        return jsonify({"error": "Model not loaded on server."}), 500
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Expecting JSON body"}), 400

    rows = None
    if "rows" in data:
        rows = np.array(data["rows"], dtype=np.float32)
    elif "row" in data:
        rows = np.array([data["row"]], dtype=np.float32)
    else:
        return jsonify({"error": "Provide 'row' or 'rows' in JSON."}), 400

    try:
        outputs = sess.run(None, {input_name: rows})
        # Convert outputs to python lists; many ONNX models return single array
        py_outs = []
        for out in outputs:
            try:
                py_outs.append(out.tolist())
            except:
                py_outs.append([float(out)])
        return jsonify({"predictions": py_outs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
