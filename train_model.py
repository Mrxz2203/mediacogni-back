"""
train_model.py — Genera datos simulados y entrena el modelo XGBoost
Ejecutar UNA SOLA VEZ con: python train_model.py
Genera: modelo_cognitivo.pkl
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

np.random.seed(42)
N = 1000  # muestras por clase

def generar_visual(n):
    return pd.DataFrame({
        'gaze_x_std':    np.random.normal(0.18, 0.03, n),   # mayor variacion horizontal
        'gaze_y_std':    np.random.normal(0.08, 0.02, n),
        'gaze_x_mean':   np.random.normal(0.52, 0.05, n),
        'gaze_y_mean':   np.random.normal(0.48, 0.04, n),
        'yaw_mean':      np.random.normal(-8,  10,   n),    # mueve cabeza horizontal
        'yaw_std':       np.random.normal(12,   3,   n),
        'pitch_mean':    np.random.normal(-5,   6,   n),
        'pitch_std':     np.random.normal(5,    2,   n),
        'roll_mean':     np.random.normal(2,    4,   n),
        'blink_mean':    np.random.normal(0.20, 0.03, n),   # parpadea menos
        'blink_std':     np.random.normal(0.04, 0.01, n),
        'pupil_mean':    np.random.normal(18,   3,   n),
        'pupil_std':     np.random.normal(3,    1,   n),
        'label': 0  # 0 = Visual
    })

def generar_verbal(n):
    return pd.DataFrame({
        'gaze_x_std':    np.random.normal(0.09, 0.02, n),
        'gaze_y_std':    np.random.normal(0.16, 0.03, n),   # mayor variacion vertical
        'gaze_x_mean':   np.random.normal(0.50, 0.04, n),
        'gaze_y_mean':   np.random.normal(0.50, 0.05, n),
        'yaw_mean':      np.random.normal(-5,   7,   n),
        'yaw_std':       np.random.normal(7,    2,   n),
        'pitch_mean':    np.random.normal(-10,  8,   n),    # mueve cabeza vertical
        'pitch_std':     np.random.normal(9,    3,   n),
        'roll_mean':     np.random.normal(3,    4,   n),
        'blink_mean':    np.random.normal(0.27, 0.04, n),   # parpadea mas
        'blink_std':     np.random.normal(0.06, 0.02, n),
        'pupil_mean':    np.random.normal(14,   3,   n),
        'pupil_std':     np.random.normal(2,    1,   n),
        'label': 1  # 1 = Verbal
    })

# Generar dataset
df = pd.concat([generar_visual(N), generar_verbal(N)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

FEATURES = [
    'gaze_x_std', 'gaze_y_std', 'gaze_x_mean', 'gaze_y_mean',
    'yaw_mean', 'yaw_std', 'pitch_mean', 'pitch_std', 'roll_mean',
    'blink_mean', 'blink_std', 'pupil_mean', 'pupil_std'
]

X = df[FEATURES]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.1f}%")
print(classification_report(y_test, y_pred, target_names=['Visual', 'Verbal']))

# Guardar modelo
with open('modelo_cognitivo.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': FEATURES}, f)

print("✅ Modelo guardado en modelo_cognitivo.pkl")