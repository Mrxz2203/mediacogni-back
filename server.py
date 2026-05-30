"""
server.py — Servidor Flask para V-COGNI
Ejecutar con: python server.py  (este solo, ya no necesitas correr eye_coordinates_cam.py aparte)
"""

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import json
import time
import threading

app = Flask(__name__)
CORS(app)

latest_gaze = {}
sse_clients = []

# ── Arrancar la cámara en un hilo interno ──────────────────────────────────
def start_camera():
    import eye_coordinates_cam
    eye_coordinates_cam.main()

camera_thread = threading.Thread(target=start_camera, daemon=True)
camera_thread.start()
# ──────────────────────────────────────────────────────────────────────────

@app.route('/gaze', methods=['POST'])
def receive_gaze():
    global latest_gaze
    latest_gaze = request.get_json()
    data = f"data: {json.dumps(latest_gaze)}\n\n"
    for client in list(sse_clients):
        try:
            client.put(data)
        except Exception:
            sse_clients.remove(client)
    return jsonify({"status": "ok"}), 200


@app.route('/stream')
def stream():
    import queue
    def event_stream():
        q = queue.Queue()
        sse_clients.append(q)
        try:
            if latest_gaze:
                yield f"data: {json.dumps(latest_gaze)}\n\n"
            while True:
                try:
                    data = q.get(timeout=30)
                    yield data
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            if q in sse_clients:
                sse_clients.remove(q)

    return Response(event_stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/video_feed')
def video_feed():
    from eye_coordinates_cam import get_current_frame

    def generate():
        while True:
            frame = get_current_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    active = bool(latest_gaze) and (time.time() * 1000 - latest_gaze.get('timestamp', 0)) < 3000
    return jsonify({"connected": active, "clients": len(sse_clients)})


if __name__ == '__main__':
    print("=" * 50)
    print("  V-COGNI server corriendo en http://localhost:5002")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5002, threaded=True)