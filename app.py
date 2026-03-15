# ================================================================
#  NeuroScan V3.0 — app.py (with Medicine Report)
# ================================================================
import os, json, datetime
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from predict import predict_tumor

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
HISTORY_FILE  = "scan_history.json"
ALLOWED       = {"png","jpg","jpeg","bmp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f: return json.load(f)
    return []

def save_history(record):
    history = load_history()
    history.insert(0, record)
    history = history[:50]
    with open(HISTORY_FILE, 'w') as f: json.dump(history, f, indent=2)

def get_stats():
    history = load_history()
    total   = len(history)
    tumors  = sum(1 for h in history if h['label'] == 'tumor')
    avg_conf = round(sum(h['confidence'] for h in history)/total, 1) if total else 0
    return {"total": total, "tumors": tumors, "normal": total-tumors, "avg_conf": avg_conf}

def allowed(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED

@app.route("/")
def home():
    return render_template("index.html", stats=get_stats(), history=load_history()[:5])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file selected.", stats=get_stats(), history=load_history()[:5])
    f = request.files["file"]
    if f.filename == "" or not allowed(f.filename):
        return render_template("index.html", error="Please upload a JPG or PNG image.", stats=get_stats(), history=load_history()[:5])
    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)
    try:
        result = predict_tumor(filepath)
        record = {
            "filename":   filename,
            "result":     result["result"],
            "label":      result["label"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "timestamp":  datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
        }
        save_history(record)
        return render_template("result.html",
            result         = result["result"],
            confidence     = result["confidence"],
            label          = result["label"],
            tumor_prob     = result["tumor_prob"],
            normal_prob    = result["normal_prob"],
            risk_level     = result["risk_level"],
            image_filename = filename,
            timestamp      = record["timestamp"],
            medicine       = result["medicine"]
        )
    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}. Run py train_model.py first!", stats=get_stats(), history=load_history()[:5])

@app.route("/history")
def history():
    return render_template("history.html", records=load_history(), stats=get_stats())

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=get_stats(), history=load_history())

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    print("\n" + "="*52)
    print("  🧠  NeuroScan V3.0 — with Medicine Report")
    print("="*52)
    print("  ✅  Open: http://127.0.0.1:5000")
    print("="*52 + "\n")
    app.run(debug=True)