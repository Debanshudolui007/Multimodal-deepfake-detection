from flask import Flask, render_template, request, jsonify
import os
import tempfile
from pathlib import Path

# Fix relative imports when running from root
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from deepfake_detector.api.predictor import DeepfakePredictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50 MB max video size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

print("Initializing DeepfakePredictor (Mock Mode)...")
# Pass the correct absolute path to the best.pth weights
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_detector", "experiments", "fusion", "checkpoints", "best.pth")
predictor = DeepfakePredictor(fusion_checkpoint_path=ckpt_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            results = predictor.predict(filepath)
            return jsonify(results)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print("PREDICTION ERROR:\n", error_trace)
            return jsonify({"error": str(e), "trace": error_trace}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    # Run the server on all interfaces
    # Turned off debug mode so watchdog doesn't interrupt active connections on file changes
    app.run(host='0.0.0.0', port=5000, debug=False)
