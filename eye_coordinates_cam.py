import os
import sys
import time
import cv2
import numpy as np
import json
import requests
import threading
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import core as mp_core

# Variable compartida para el frame actual (thread-safe)
current_frame = None
frame_lock = threading.Lock()

def get_model_path():
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", "face_landmarker_v2.task")

# Índices de MediaPipe Face Mesh para ojos, iris y párpados
LEFT_EYE = (33, 133)
RIGHT_EYE = (362, 263)
# Face Mesh extendido con iris: 5 landmarks por ojo
RIGHT_IRIS = (468, 469, 470, 471, 472)
LEFT_IRIS = (473, 474, 475, 476, 477)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
HEAD_POSE_LANDMARKS = [33, 263, 1, 199, 61, 291]

# Modelo 3D aproximado para la pose de la cabeza (solución clásica solvePnP)
FACE_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],        # nariz
        [0.0, -63.6, -12.5],    # barbilla
        [-43.3, 32.7, -26.0],   # ojo izquierdo externo
        [43.3, 32.7, -26.0],    # ojo derecho externo
        [-28.9, -28.9, -20.0],  # comisura izquierda boca
        [28.9, -28.9, -20.0],   # comisura derecha boca
    ],
    dtype=np.float64,
)


def landmark_to_point(landmark, image_width, image_height):
    return int(landmark.x * image_width), int(landmark.y * image_height)


def get_average_landmark(landmarks, indices):
    x = sum(landmarks[i].x for i in indices) / len(indices)
    y = sum(landmarks[i].y for i in indices) / len(indices)
    return x, y


def compute_iris_center_and_diameter(landmarks, iris_indices, image_width, image_height):
    points = np.array(
        [[landmarks[i].x * image_width, landmarks[i].y * image_height] for i in iris_indices],
        dtype=np.float64,
    )
    center = np.mean(points, axis=0)
    radius = np.mean(np.linalg.norm(points - center, axis=1))
    return (int(center[0]), int(center[1])), float(radius * 2), int(radius)


def compute_eye_aspect_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx, image_width, image_height):
    top = np.array([landmarks[top_idx].x * image_width, landmarks[top_idx].y * image_height])
    bottom = np.array([landmarks[bottom_idx].x * image_width, landmarks[bottom_idx].y * image_height])
    left = np.array([landmarks[left_idx].x * image_width, landmarks[left_idx].y * image_height])
    right = np.array([landmarks[right_idx].x * image_width, landmarks[right_idx].y * image_height])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def estimate_distance_to_screen(iris_diameter_px, focal_length_px, average_iris_diameter_mm=11.8):
    if iris_diameter_px <= 0:
        return None
    return (average_iris_diameter_mm * focal_length_px) / iris_diameter_px


def compute_relative_eye_movement(eye_coords, iris_point):
    eye_center = ((eye_coords[0][0] + eye_coords[1][0]) // 2, (eye_coords[0][1] + eye_coords[1][1]) // 2)
    rel_x = iris_point[0] - eye_center[0]
    rel_y = iris_point[1] - eye_center[1]
    width = max(abs(eye_coords[1][0] - eye_coords[0][0]), 1)
    height = max(abs(eye_coords[1][1] - eye_coords[0][1]), 1)
    return rel_x / width, rel_y / height


def estimate_head_pose(landmarks, image_width, image_height):
    if max(HEAD_POSE_LANDMARKS) >= len(landmarks):
        return None

    image_points = np.array(
        [
            [landmarks[i].x * image_width, landmarks[i].y * image_height]
            for i in HEAD_POSE_LANDMARKS
        ],
        dtype=np.float64,
    )

    focal_length = image_width
    camera_matrix = np.array(
        [[focal_length, 0, image_width / 2], [0, focal_length, image_height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy < 1e-6:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    pitch = np.degrees(x)
    yaw = np.degrees(y)
    roll = np.degrees(z)
    return yaw, pitch, roll


def create_image_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp_core.image.Image(
        mp_core.image.ImageFormat.SRGB, rgb_frame
    )
    return image


def draw_face_mesh(frame, landmarks, image_width, image_height):
    def point(index):
        return (
            int(landmarks[index].x * image_width),
            int(landmarks[index].y * image_height),
        )

    for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        if connection.start < len(landmarks) and connection.end < len(landmarks):
            cv2.line(frame, point(connection.start), point(connection.end), (255, 128, 0), 1)

    for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
        if connection.start < len(landmarks) and connection.end < len(landmarks):
            cv2.line(frame, point(connection.start), point(connection.end), (0, 255, 0), 2)

    for idx in RIGHT_IRIS + LEFT_IRIS:
        if idx < len(landmarks):
            cv2.circle(frame, point(idx), 2, (0, 255, 255), -1)


def send_gaze_data(gaze_data):
    """Envía datos de gaze al servidor HTTP en un thread separado."""
    try:
        requests.post("http://localhost:5002/gaze", json=gaze_data, timeout=0.1)
    except Exception as e:
        pass  # No bloquea el pipeline de video


def main():
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo. Descarga el archivo en: {model_path}"
        )

    print(f"Cargando modelo desde: {model_path}")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Iniciando cámara...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presione ESC para salir.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            mp_image = create_image_from_frame(frame)
            timestamp_ms = int(time.time() * 1000)

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    landmarks = face_landmarks
                    left_eye_coords = [
                        landmark_to_point(landmarks[i], frame_width, frame_height)
                        for i in LEFT_EYE
                    ]
                    right_eye_coords = [
                        landmark_to_point(landmarks[i], frame_width, frame_height)
                        for i in RIGHT_EYE
                    ]

                    if max(LEFT_IRIS + RIGHT_IRIS) < len(landmarks):
                        left_iris_point, left_iris_diameter, left_iris_radius = compute_iris_center_and_diameter(
                            landmarks, LEFT_IRIS, frame_width, frame_height
                        )
                        right_iris_point, right_iris_diameter, right_iris_radius = compute_iris_center_and_diameter(
                            landmarks, RIGHT_IRIS, frame_width, frame_height
                        )
                        for idx in RIGHT_IRIS + LEFT_IRIS:
                            cv2.circle(
                                frame,
                                landmark_to_point(landmarks[idx], frame_width, frame_height),
                                2,
                                (0, 255, 255),
                                -1,
                            )
                    else:
                        # Si el modelo Iris no está disponible, usamos el centro del ojo como aproximación.
                        left_iris_point = (
                            (left_eye_coords[0][0] + left_eye_coords[1][0]) // 2,
                            (left_eye_coords[0][1] + left_eye_coords[1][1]) // 2,
                        )
                        right_iris_point = (
                            (right_eye_coords[0][0] + right_eye_coords[1][0]) // 2,
                            (right_eye_coords[0][1] + right_eye_coords[1][1]) // 2,
                        )
                        left_iris_diameter = np.linalg.norm(
                            np.array(left_eye_coords[0]) - np.array(left_eye_coords[1])
                        )
                        right_iris_diameter = np.linalg.norm(
                            np.array(right_eye_coords[0]) - np.array(right_eye_coords[1])
                        )
                        left_iris_radius = right_iris_radius = int(min(left_iris_diameter, right_iris_diameter) / 4)

                    left_blink_ratio = compute_eye_aspect_ratio(
                        landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE[0], LEFT_EYE[1], frame_width, frame_height
                    )
                    right_blink_ratio = compute_eye_aspect_ratio(
                        landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE[0], RIGHT_EYE[1], frame_width, frame_height
                    )
                    left_eye_movement = compute_relative_eye_movement(left_eye_coords, left_iris_point)
                    right_eye_movement = compute_relative_eye_movement(right_eye_coords, right_iris_point)

                    distance_left_mm = estimate_distance_to_screen(left_iris_diameter, frame_width)
                    distance_right_mm = estimate_distance_to_screen(right_iris_diameter, frame_width)
                    distance_text = f"Dist L:{distance_left_mm:.0f}mm R:{distance_right_mm:.0f}mm" if distance_left_mm and distance_right_mm else "Dist: N/A"

                    draw_face_mesh(frame, landmarks, frame_width, frame_height)

                    for coord in left_eye_coords + right_eye_coords:
                        cv2.circle(frame, coord, 3, (0, 0, 255), -1)

                    cv2.circle(frame, left_iris_point, left_iris_radius, (255, 0, 0), 1)
                    cv2.circle(frame, right_iris_point, right_iris_radius, (255, 0, 0), 1)
                    cv2.circle(frame, left_iris_point, 4, (255, 0, 0), -1)
                    cv2.circle(frame, right_iris_point, 4, (255, 0, 0), -1)

                    pose = estimate_head_pose(landmarks, frame_width, frame_height)
                    if pose is not None:
                        yaw, pitch, roll = pose
                        cv2.putText(
                            frame,
                            f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                        print(f"Head pose (deg): yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")

                    blink_left = left_blink_ratio < 0.22
                    blink_right = right_blink_ratio < 0.22
                    blink_text = (
                        f"Blink L:{'Y' if blink_left else 'N'} R:{'Y' if blink_right else 'N'}"
                    )
                    movement_text = (
                        f"EyeMov L:({left_eye_movement[0]:.2f},{left_eye_movement[1]:.2f}) "
                        f"R:({right_eye_movement[0]:.2f},{right_eye_movement[1]:.2f})"
                    )

                    cv2.putText(frame, distance_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(frame, blink_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(frame, movement_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    # Enviar datos de gaze al servidor HTTP/WebSocket
                    gaze_data = {
                        "timestamp": timestamp_ms,
                        "gazeXNorm": (left_iris_point[0] + right_iris_point[0]) / (2 * frame_width),
                        "gazeYNorm": (left_iris_point[1] + right_iris_point[1]) / (2 * frame_height),
                        "blinkRatio": min(left_blink_ratio, right_blink_ratio),
                        "pupilDiameterPx": (left_iris_diameter + right_iris_diameter) / 2,
                        "headPose": {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)} if pose else None,
                    }
                    threading.Thread(target=send_gaze_data, args=(gaze_data,), daemon=True).start()

                    print(
                        f"Ojo izquierdo: corner1={left_eye_coords[0]}, corner2={left_eye_coords[1]}, iris={left_iris_point}, blink_ratio={left_blink_ratio:.2f}"
                    )
                    print(
                        f"Ojo derecho: corner1={right_eye_coords[0]}, corner2={right_eye_coords[1]}, iris={right_iris_point}, blink_ratio={right_blink_ratio:.2f}"
                    )

            # Mostrar el frame procesado y permitir salida con ESC
            # cv2.imshow('Eye Coordinates', frame)
            if cv2.waitKey(1) == 27:
                print('Saliendo por tecla ESC.')
                break

            # Guardar frame para MJPEG streaming
            global current_frame
            with frame_lock:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    current_frame = jpeg.tobytes()

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def get_current_frame():
    """Devuelve el frame actual para MJPEG streaming."""
    global current_frame
    with frame_lock:
        return current_frame


if __name__ == "__main__":
    main()
