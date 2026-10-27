TECHGUARD WEB — UPGRADED SCHOOL SAFETY PROTOTYPE
================================================

Features added:
- Sign-in and sign-up pages with Werkzeug password hashing and Flask sessions.
- First registered account becomes admin; later signups become staff.
- SQLite camera directory with camera code, building, floor, room and directions.
- Location-tagged AI incident queue with acknowledge / resolve actions.
- Protected dashboard and API routes.
- Optional private Hugging Face model download at startup.

IMPORTANT BEFORE DEPLOYING
1. Set a long random SECRET_KEY in Render Environment.
2. Keep TG_USERNAME/TG_PASSWORD out of use for this version; the app uses its own account sign-in.
3. Set HF_MODEL_REPO=ambatokum/yolov
4. Set HF_TOKEN to a Hugging Face read token (keep secret).
5. Set HF_YOLO_FILENAME and HF_RESNET_FILENAME to the exact filenames shown in Hugging Face.
6. Set persistent storage for techguard.db if you need accounts, camera records and incidents to survive redeploys/restarts. Without a persistent disk/database service, local SQLite data may be lost.
7. Sign up the first account before sharing the URL. The first account becomes admin; later self-registrations become staff. For real school deployment, disable open registration or require an admin invitation.
8. This is a prototype, not production-ready security software. Add CSRF protection, rate limiting, audit/retention controls, backups and managed persistent database before real school use.

CAMERAS
Camera directory entries store location information. They do not magically connect physical CCTV. Current camera source selection uses OpenCV device indices, which usually work on a machine directly attached to the camera, not on Render. For real remote CCTV, add a secure server-side RTSP/NVR integration or a camera gateway. Do not expose camera credentials in browser JavaScript.

AI MODEL FILES
If HF_MODEL_REPO is set, the app downloads files from that private Hugging Face repository using HF_TOKEN. Ensure both exact filenames exist. Otherwise it looks for models/yolov8n.pt and models/violence_resnet18.pth.

LOCAL RUN
pip install -r requirements.txt
python app.py
Then visit http://127.0.0.1:5000 and create the first account.

AI SAFETY
AI flags are review signals, not proof of violence or misconduct. An authorised human must verify incidents. Process only footage you are authorised to use and follow school privacy and retention rules.
