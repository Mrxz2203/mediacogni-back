"""
classifier.py — Carga el modelo y clasifica métricas de una sesión
"""

import pickle
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_cognitivo.pkl')
_model_cache = None

def cargar_modelo():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, 'rb') as f:
            _model_cache = pickle.load(f)
    return _model_cache

def clasificar(metricas: list[dict]) -> dict:
    """
    Recibe lista de métricas gaze capturadas durante la sesión.
    Cada métrica: {yaw, pitch, roll, gaze_x, gaze_y, blink_ratio, pupil_px, gaze_zone}
    Retorna: {estilo: 'Visual'|'Verbal', confianza: float}
    """
    if not metricas or len(metricas) < 5:
        return {"estilo": "Indeterminado", "confianza": 0.0}

    gaze_x = [m.get('gaze_x', 0.5)      for m in metricas]
    gaze_y = [m.get('gaze_y', 0.5)      for m in metricas]
    yaw    = [m.get('yaw', 0)            for m in metricas]
    pitch  = [m.get('pitch', 0)          for m in metricas]
    roll   = [m.get('roll', 0)           for m in metricas]
    blink  = [m.get('blink_ratio', 0.25) for m in metricas]
    pupil  = [m.get('pupil_px', 15)      for m in metricas]

    # AOI: % de frames en zona imagen (gaze_zone == 'image')
    # Si la métrica no trae gaze_zone (sesiones antiguas), se estima por gaze_x > 0.5
    zonas      = [m.get('gaze_zone') for m in metricas]
    tiene_zona = any(z is not None for z in zonas)
    if tiene_zona:
        n_image = sum(1 for z in zonas if z == 'image')
    else:
        # Fallback para sesiones capturadas antes del AOI tracking
        n_image = sum(1 for x in gaze_x if x > 0.5)
    pct_image = n_image / len(metricas)

    # El orden debe coincidir exactamente con FEATURES en train_model.py
    features = [[
        float(np.std(gaze_x)),
        float(np.std(gaze_y)),
        float(np.mean(gaze_x)),
        float(np.mean(gaze_y)),
        float(np.mean(yaw)),
        float(np.std(yaw)),
        float(np.mean(pitch)),
        float(np.std(pitch)),
        float(np.mean(roll)),
        float(np.mean(blink)),
        float(np.std(blink)),
        float(np.mean(pupil)),
        float(np.std(pupil)),
        float(pct_image),          # ← AOI feature
    ]]

    data  = cargar_modelo()
    model = data['model']

    # Validación: confirmar que el modelo fue entrenado con el mismo nº de features
    expected = len(data.get('features', []))
    if expected and len(features[0]) != expected:
        raise ValueError(
            f"Feature mismatch: el clasificador generó {len(features[0])} features "
            f"pero el modelo espera {expected}. Regenera modelo_cognitivo.pkl "
            f"ejecutando train_model.py."
        )

    proba     = model.predict_proba(features)[0]
    pred      = model.predict(features)[0]
    estilo    = 'Visual' if pred == 0 else 'Verbal'
    confianza = float(proba[pred])

    return {
        "estilo":     estilo,
        "confianza":  round(confianza, 3),
        "pct_image":  round(pct_image, 3),   # devuelto para logging/debug en api.py
    }