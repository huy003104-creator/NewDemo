from __future__ import annotations

import datetime
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, session, flash
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
EVIDENCE_DIR = BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

YOLO_PATH = MODEL_DIR / "yolov8n.pt"
RESNET_PATH = MODEL_DIR / "violence_resnet18.pth"

FLOW_THRESHOLD = 0.45
MOVE_THRESHOLD = 35
CONFIRM_SECONDS = 3
BUFFER_SIZE = 15
STAGE2_THRESHOLD = 0.85
DETECTION_EVERY_N_FRAMES = 3

app = Flask(__name__)

# Session authentication and staff accounts.
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or os.urandom(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(os.getenv("RENDER"))
DB_PATH = Path(os.getenv("TECHGUARD_DB", str(BASE_DIR / "techguard.db")))

def db():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'staff',
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            building TEXT NOT NULL,
            floor TEXT NOT NULL,
            room TEXT NOT NULL,
            details TEXT DEFAULT '',
            source_index INTEGER DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_code TEXT NOT NULL,
            location TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'Review',
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open',
            created_at TEXT NOT NULL,
            acknowledged_by TEXT DEFAULT '',
            resolved_by TEXT DEFAULT '',
            evidence_path TEXT DEFAULT ''
        )""")
        count = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        if count == 0:
            now = datetime.datetime.now().isoformat(timespec="seconds")
            conn.executemany("""INSERT INTO cameras
                (camera_code,name,building,floor,room,details,source_index,active,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""", [
                ("CAM-01","Main entrance","Main Building","Ground","Entrance","Front gate",0,1,now),
                ("CAM-02","North corridor","Main Building","2","Corridor N2","Demo camera; configure its source",1,1,now),
                ("CAM-03","Science block","Science Building","1","Lab corridor","Demo camera; configure its source",2,1,now),
            ])
    # Werkzeug's password hashing is part of Flask's dependency.
init_db()

@app.before_request
def require_login():
    public_endpoints = {"login", "login_post", "signup", "signup_post", "static"}
    if request.endpoint in public_endpoints or request.endpoint is None:
        return None
    if not session.get("user_id"):
        if request.path.startswith("/api/") or request.path == "/video-feed":
            return jsonify({"ok": False, "error": "Please sign in first."}), 401
        return redirect(url_for("login", next=request.path))
    return None

@app.get("/login")
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("index.html", auth_mode="login")

@app.post("/login")
def login_post():
    from werkzeug.security import check_password_hash
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "")
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        flash("Incorrect username or password.", "error")
        return redirect(url_for("login"))
    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]
    return redirect(url_for("index"))

@app.get("/signup")
def signup():
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("index.html", auth_mode="signup")

@app.post("/signup")
def signup_post():
    from werkzeug.security import generate_password_hash
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")
    if len(username) < 3 or len(username) > 40 or not all(c.isalnum() or c in "._-" for c in username):
        flash("Username must be 3–40 characters and use letters, numbers, dots, underscores or hyphens.", "error")
        return redirect(url_for("signup"))
    if len(password) < 10:
        flash("Use a password with at least 10 characters.", "error")
        return redirect(url_for("signup"))
    if password != confirm:
        flash("Passwords do not match.", "error")
        return redirect(url_for("signup"))
    with db() as conn:
        try:
            existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            # First account becomes admin; subsequent signups are staff.
            role = "admin" if existing == 0 else "staff"
            conn.execute("INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)",
                         (username, generate_password_hash(password), role, datetime.datetime.now().isoformat(timespec="seconds")))
        except Exception:
            flash("That username is already registered.", "error")
            return redirect(url_for("signup"))
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]
    return redirect(url_for("index"))

@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

state_lock = threading.Lock()
state = {
    "running": False,
    "source": "none",
    "status": "READY",
    "confidence": 0.0,
    "flow": 0.0,
    "fight_score": 0,
    "people": 0,
    "events": [],
    "error": None,
    "camera_code": "CAM-01",
    "location": "Main Building · Ground · Entrance",
}
cap = None
worker = None
latest_jpeg = None
stop_event = threading.Event()

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = None
stage2 = None
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def log_event(message, level="info"):
    item = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "level": level,
    }
    with state_lock:
        item["camera_code"] = state.get("camera_code", "CAM-01")
        item["location"] = state.get("location", "Unspecified location")
        state["events"].insert(0, item)
        del state["events"][100:]
    if level == "alert":
        try:
            with db() as conn:
                conn.execute("""INSERT INTO incidents(camera_code,location,severity,description,status,created_at,evidence_path)
                    VALUES(?,?,?,?,?,?,?)""",
                    (item["camera_code"], item["location"], "Urgent review", message,
                     "Open", datetime.datetime.now().isoformat(timespec="seconds"), ""))
        except Exception as exc:
            print("Could not persist incident:", exc)

def load_models():
    global yolo, stage2
    yolo_path = YOLO_PATH
    resnet_path = RESNET_PATH
    repo_id = os.getenv("HF_MODEL_REPO")
    hf_token = os.getenv("HF_TOKEN")
    if repo_id:
        from huggingface_hub import hf_hub_download
        yolo_name = os.getenv("HF_YOLO_FILENAME", "yolov8n.pt")
        resnet_name = os.getenv("HF_RESNET_FILENAME", "violence_resnet18.pth")
        yolo_path = Path(hf_hub_download(repo_id=repo_id, filename=yolo_name, token=hf_token))
        resnet_path = Path(hf_hub_download(repo_id=repo_id, filename=resnet_name, token=hf_token))
    if not yolo_path.is_file():
        raise FileNotFoundError(f"Missing YOLO model at {yolo_path}. Set HF_MODEL_REPO/HF_YOLO_FILENAME or add models/yolov8n.pt.")
    if not resnet_path.is_file():
        raise FileNotFoundError(f"Missing ResNet model at {resnet_path}. Set HF_MODEL_REPO/HF_RESNET_FILENAME or add models/violence_resnet18.pth.")
    log_event("Loading YOLOv8 model…")
    yolo = YOLO(str(yolo_path))
    try:
        yolo.fuse()
    except Exception:
        pass
    log_event("YOLOv8 loaded.")
    log_event("Loading ResNet18 classifier…")
    stage2 = models.resnet18(weights=None)
    stage2.fc = nn.Linear(stage2.fc.in_features, 2)
    weights = torch.load(str(resnet_path), map_location=device, weights_only=True)
    stage2.load_state_dict(weights)
    stage2.to(device)
    stage2.eval()
    log_event(f"ResNet18 loaded on {device}.")
    with state_lock:
        state["status"] = "READY"

def stage2_predict(frames):
    if not frames:
        return 0.0
    probs = []
    with torch.inference_mode():
        for frame in frames:
            # OpenCV frames are BGR; convert to RGB for the classifier.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(device)
            output = stage2(tensor)
            probability = torch.softmax(output, dim=1)
            probs.append(float(probability[0][1].item()))
    return float(np.mean(probs))

def camera_worker(source, is_file):
    global cap, latest_jpeg
    prev_gray = None
    prev_centers = []
    buffer = deque(maxlen=BUFFER_SIZE)
    suspicious_since = None
    last_alert = 0
    frame_count = 0
    cached_boxes = []
    cached_flow = 0.0
    cached_score = 0
    cached_stage1 = False
    cached_confidence = 0.0
    cached_violent = False

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    log_event("Test video finished.")
                else:
                    log_event("Camera frame could not be read.", "warning")
                break

            if frame.shape[1] > 960:
                scale = 960 / frame.shape[1]
                frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

            buffer.append(frame.copy())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            if frame_count % DETECTION_EVERY_N_FRAMES == 0 or frame_count == 1:
                results = yolo.predict(source=frame, conf=0.4, classes=[0], verbose=False, imgsz=640)
                centers, boxes = [], []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
                        boxes.append((x1, y1, x2, y2))

                score = 0
                for i in range(min(len(centers), len(prev_centers))):
                    dx = centers[i][0] - prev_centers[i][0]
                    dy = centers[i][1] - prev_centers[i][1]
                    if (dx * dx + dy * dy) ** 0.5 > MOVE_THRESHOLD:
                        score += 1

                flow_energy = 0.0
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_energy = float(np.mean(magnitude))

                stage1 = flow_energy > FLOW_THRESHOLD or score >= 2
                confidence = 0.0
                violent = False
                now = time.time()
                if stage1:
                    if suspicious_since is None:
                        suspicious_since = now
                        log_event("Suspicious movement; running confirmation stage.")
                    elif now - suspicious_since >= CONFIRM_SECONDS and len(buffer) == BUFFER_SIZE:
                        confidence = stage2_predict(list(buffer))
                        if confidence > STAGE2_THRESHOLD:
                            violent = True
                else:
                    suspicious_since = None

                cached_boxes = boxes
                cached_flow = flow_energy
                cached_score = score
                cached_stage1 = stage1
                cached_confidence = confidence
                cached_violent = violent
                prev_gray = gray.copy()
                prev_centers = centers.copy()

            people = len(cached_boxes)
            if cached_violent:
                status_text = "POSSIBLE VIOLENCE - REVIEW"
                color = (0, 0, 255)
                if time.time() - last_alert > 2:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    evidence_path = EVIDENCE_DIR / f"alert_{stamp}.jpg"
                    cv2.imwrite(str(evidence_path), frame)
                    log_event(f"Possible violence flagged; evidence saved: {evidence_path.name}", "alert")
                    last_alert = time.time()
            elif cached_stage1:
                status_text = "CHECKING - REVIEW"
                color = (0, 165, 255)
            else:
                status_text = "MONITORING"
                color = (0, 200, 0)

            for x1, y1, x2, y2 in cached_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status_text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"People: {people} | Violence score: {cached_confidence:.2f}", (18, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with state_lock:
                    latest_jpeg = encoded.tobytes()
                    state["people"] = people
                    state["confidence"] = round(cached_confidence, 3)
                    state["flow"] = round(cached_flow, 3)
                    state["fight_score"] = cached_score
                    state["status"] = status_text

            # Limit file playback from overwhelming the browser; camera feed remains responsive.
            time.sleep(0.01)
    except Exception as exc:
        log_event(f"Processing error: {exc}", "error")
        with state_lock:
            state["error"] = str(exc)
    finally:
        if cap is not None:
            cap.release()
        with state_lock:
            cap = None
            state["running"] = False
            if state["status"] not in ("POSSIBLE VIOLENCE - REVIEW", "CHECKING - REVIEW"):
                state["status"] = "STOPPED"
        log_event("Monitoring stopped.")

def start_monitoring(source, is_file=False):
    global cap, worker
    if yolo is None or stage2 is None:
        raise RuntimeError("AI models are not loaded. Check the startup log.")
    with state_lock:
        if state["running"]:
            raise RuntimeError("Monitoring is already running.")
    stop_event.clear()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        cap = None
        raise RuntimeError("Could not open the selected camera or video.")
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    with state_lock:
        state["running"] = True
        state["source"] = "video" if is_file else f"camera {source}"
        state["error"] = None
    log_event(f"Monitoring started: {state['source']}.")
    worker = threading.Thread(target=camera_worker, args=(source, is_file), daemon=True)
    worker.start()

@app.get("/")
def index():
    return render_template("index.html", auth_mode=None, username=session.get("username"), role=session.get("role"))

@app.get("/api/status")
def api_status():
    with state_lock:
        snapshot = {**state, "events": list(state["events"])}
    with db() as conn:
        snapshot["cameras"] = [dict(r) for r in conn.execute("SELECT * FROM cameras ORDER BY camera_code")]
        snapshot["incidents"] = [dict(r) for r in conn.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT 50")]
    return jsonify(snapshot)

@app.get("/api/cameras")
def api_cameras():
    with db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM cameras ORDER BY camera_code")]
    return jsonify({"ok": True, "cameras": rows})

@app.post("/api/cameras")
def api_add_camera():
    data = request.get_json(silent=True) or {}
    required = ["camera_code", "name", "building", "floor", "room"]
    if any(not str(data.get(k, "")).strip() for k in required):
        return jsonify({"ok": False, "error": "Camera code, name, building, floor and room are required."}), 400
    try:
        with db() as conn:
            conn.execute("""INSERT INTO cameras(camera_code,name,building,floor,room,details,source_index,active,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""", (
                str(data["camera_code"]).strip().upper(), str(data["name"]).strip(),
                str(data["building"]).strip(), str(data["floor"]).strip(), str(data["room"]).strip(),
                str(data.get("details","")).strip(), int(data.get("source_index",0)), 1,
                datetime.datetime.now().isoformat(timespec="seconds")))
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False, "error": "Camera code already exists or the details are invalid."}), 400

@app.post("/api/start-camera")
def api_start_camera():
    data = request.get_json(silent=True) or {}
    try:
        camera_id = int(data.get("source_index", 0))
        with db() as conn:
            camera = conn.execute("SELECT * FROM cameras WHERE camera_code=? AND active=1",
                                  (str(data.get("camera_code", "CAM-01")).upper(),)).fetchone()
        if not camera:
            return jsonify({"ok": False, "error": "Choose an active camera."}), 400
        with state_lock:
            state["camera_code"] = camera["camera_code"]
            state["location"] = f'{camera["building"]} · Floor {camera["floor"]} · {camera["room"]}'
        start_monitoring(camera_id, is_file=False)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.post("/api/start-video")
def api_start_video():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Choose a video file first."}), 400
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        return jsonify({"ok": False, "error": "Supported formats: MP4, AVI, MOV, MKV."}), 400
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    saved = upload_dir / f"test_{datetime.datetime.now():%Y%m%d_%H%M%S}{suffix}"
    file.save(saved)
    data = request.form
    camera_code = str(data.get("camera_code", "CAM-01")).upper()
    with db() as conn:
        camera = conn.execute("SELECT * FROM cameras WHERE camera_code=?", (camera_code,)).fetchone()
    if camera:
        with state_lock:
            state["camera_code"] = camera["camera_code"]
            state["location"] = f'{camera["building"]} · Floor {camera["floor"]} · {camera["room"]}'
    try:
        start_monitoring(str(saved), is_file=True)
        return jsonify({"ok": True, "filename": saved.name})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.post("/api/stop")
def api_stop():
    stop_event.set()
    with state_lock:
        running = state["running"]
    if not running:
        return jsonify({"ok": True, "message": "Already stopped."})
    return jsonify({"ok": True, "message": "Stop requested."})

@app.post("/api/incidents/<int:incident_id>/status")
def api_incident_status(incident_id):
    data = request.get_json(silent=True) or {}
    status_value = data.get("status")
    if status_value not in {"Open", "Acknowledged", "Resolved"}:
        return jsonify({"ok": False, "error": "Invalid incident status."}), 400
    username = session.get("username", "staff")
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        row = conn.execute("SELECT id FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not row:
            return jsonify({"ok": False, "error": "Incident not found."}), 404
        if status_value == "Acknowledged":
            conn.execute("UPDATE incidents SET status=?, acknowledged_by=? WHERE id=?", (status_value, username, incident_id))
        elif status_value == "Resolved":
            conn.execute("UPDATE incidents SET status=?, resolved_by=? WHERE id=?", (status_value, username, incident_id))
        else:
            conn.execute("UPDATE incidents SET status=? WHERE id=?", (status_value, incident_id))
    log_event(f"Incident #{incident_id} marked {status_value} by {username}.")
    return jsonify({"ok": True})

@app.get("/video-feed")
def video_feed():
    def generate():
        while True:
            with state_lock:
                frame = latest_jpeg
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Load models at import time so production WSGI servers (Gunicorn) load them too.
try:
    load_models()
except Exception as exc:
    log_event(f"Model loading failed: {exc}", "error")
    with state_lock:
        state["error"] = str(exc)
        state["status"] = "MODEL ERROR"
    print(f"MODEL ERROR: {exc}")
    print("Ensure both model files are available in the models folder.")

if __name__ == "__main__":
    # Local development only. Hosted deployment uses Gunicorn and the platform port.
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
