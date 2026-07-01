"""
train_model.py — Genera datos simulados y entrena el modelo XGBoost
Ejecutar UNA SOLA VEZ con: python train_model.py
Genera: modelo_cognitivo.pkl
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import pickle

np.random.seed(42)
N = 1000  # muestras por clase

def generar_visual(n):
    return pd.DataFrame({
        'gaze_x_std':  np.random.normal(0.18, 0.03, n),   # mayor variación horizontal
        'gaze_y_std':  np.random.normal(0.08, 0.02, n),
        'gaze_x_mean': np.random.normal(0.52, 0.05, n),
        'gaze_y_mean': np.random.normal(0.48, 0.04, n),
        'yaw_mean':    np.random.normal(-8,  10,    n),    # mueve cabeza horizontal
        'yaw_std':     np.random.normal(12,   3,    n),
        'pitch_mean':  np.random.normal(-5,   6,    n),
        'pitch_std':   np.random.normal(5,    2,    n),
        'roll_mean':   np.random.normal(2,    4,    n),
        'blink_mean':  np.random.normal(0.20, 0.03, n),   # parpadea menos
        'blink_std':   np.random.normal(0.04, 0.01, n),
        'pupil_mean':  np.random.normal(18,   3,    n),
        'pupil_std':   np.random.normal(3,    1,    n),
        # Visual → más tiempo mirando la imagen (zona derecha)
        'pct_image':   np.clip(np.random.normal(0.70, 0.10, n), 0, 1),
        'label': 0    # 0 = Visual
    })

def generar_verbal(n):
    return pd.DataFrame({
        'gaze_x_std':  np.random.normal(0.09, 0.02, n),
        'gaze_y_std':  np.random.normal(0.16, 0.03, n),   # mayor variación vertical
        'gaze_x_mean': np.random.normal(0.50, 0.04, n),
        'gaze_y_mean': np.random.normal(0.50, 0.05, n),
        'yaw_mean':    np.random.normal(-5,   7,    n),
        'yaw_std':     np.random.normal(7,    2,    n),
        'pitch_mean':  np.random.normal(-10,  8,    n),   # mueve cabeza vertical
        'pitch_std':   np.random.normal(9,    3,    n),
        'roll_mean':   np.random.normal(3,    4,    n),
        'blink_mean':  np.random.normal(0.27, 0.04, n),   # parpadea más
        'blink_std':   np.random.normal(0.06, 0.02, n),
        'pupil_mean':  np.random.normal(14,   3,    n),
        'pupil_std':   np.random.normal(2,    1,    n),
        # Verbal → más tiempo mirando el texto (zona izquierda)
        'pct_image':   np.clip(np.random.normal(0.30, 0.10, n), 0, 1),
        'label': 1    # 1 = Verbal
    })

# ── Dataset ──────────────────────────────────────────────────────────────────
df = pd.concat([generar_visual(N), generar_verbal(N)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

FEATURES = [
    'gaze_x_std', 'gaze_y_std', 'gaze_x_mean', 'gaze_y_mean',
    'yaw_mean', 'yaw_std', 'pitch_mean', 'pitch_std', 'roll_mean',
    'blink_mean', 'blink_std', 'pupil_mean', 'pupil_std',
    'pct_image',   # ← AOI: % de frames en zona imagen (diagrama)
]

X = df[FEATURES]
y = df['label']

# ── Cross-validation 5 folds (accuracy más honesto que un split único) ───────
model_cv = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_cv, X, y, cv=cv, scoring='accuracy')

print("── Validación cruzada (5 folds) ──────────────────────────────")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score * 100:.1f}%")
print(f"   Media:  {cv_scores.mean() * 100:.1f}%  ±  {cv_scores.std() * 100:.1f}%")

# ── Entrenamiento final sobre el dataset completo (split 80/20 para reporte) ─
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print("\n── Evaluación en test set (20%) ───────────────────────────────")
print(f"   Accuracy: {acc * 100:.1f}%")
print(classification_report(y_test, y_pred, target_names=['Visual', 'Verbal']))

# ── Feature importance ────────────────────────────────────────────────────────
print("── Importancia de features ────────────────────────────────────")
importances = model.feature_importances_
ranked = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
for feat, imp in ranked:
    bar = '█' * int(imp * 40)
    print(f"   {feat:<16} {bar} {imp:.3f}")

# ── Guardar modelo ────────────────────────────────────────────────────────────
with open('modelo_cognitivo.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': FEATURES}, f)

print("\n✅ Modelo guardado en modelo_cognitivo.pkl")
print(f"   Features: {len(FEATURES)}  |  CV accuracy: {cv_scores.mean()*100:.1f}%  |  Test accuracy: {acc*100:.1f}%")