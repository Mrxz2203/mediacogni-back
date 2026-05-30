"""
api.py — FastAPI para V-COGNI
Ejecutar con: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

import models, schemas
from database import engine, get_db
from classifier import clasificar

# ── Crear tablas ───────────────────────────────────
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="V-COGNI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── JWT ────────────────────────────────────────────
SECRET_KEY         = "vcogni_secret_key_2026"
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p):   return pwd_context.hash(p)
def verify_password(p, h): return pwd_context.verify(p, h)

def create_token(data):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        codigo  = payload.get("sub")
        if not codigo:
            raise HTTPException(status_code=401, detail="Token inválido")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")
    user = db.query(models.Usuario).filter(models.Usuario.codigo == codigo).first()
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    return user


# ── Schema para clasificar ─────────────────────────
class MetricaGaze(BaseModel):
    yaw:         float
    pitch:       float
    roll:        float
    gaze_x:      float
    gaze_y:      float
    blink_ratio: float
    pupil_px:    float

class ClasificarRequest(BaseModel):
    metricas: List[MetricaGaze]
    duracion: Optional[int] = 90


# ── ENDPOINTS ──────────────────────────────────────

@app.get("/")
def root():
    return {"message": "V-COGNI API corriendo ✓"}


# Register
@app.post("/auth/register", response_model=schemas.UsuarioOut, status_code=201)
def register(data: schemas.RegisterSchema, db: Session = Depends(get_db)):
    existing = db.query(models.Usuario).filter(models.Usuario.codigo == data.codigo).first()
    if existing:
        raise HTTPException(status_code=400, detail="El código ya está registrado")
    user = models.Usuario(
        nombre   = data.nombre,
        codigo   = data.codigo,
        password = hash_password(data.password),
        rol      = data.rol,
        carrera  = data.carrera,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# Login
@app.post("/auth/login", response_model=schemas.TokenSchema)
def login(data: schemas.LoginSchema, db: Session = Depends(get_db)):
    user = db.query(models.Usuario).filter(models.Usuario.codigo == data.codigo).first()
    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Código o contraseña incorrectos")
    token = create_token({"sub": user.codigo, "rol": user.rol})
    return {"access_token": token, "token_type": "bearer", "rol": user.rol, "nombre": user.nombre}


# Perfil
@app.get("/usuarios/me", response_model=schemas.UsuarioOut)
def get_me(token: str, db: Session = Depends(get_db)):
    return get_current_user(token, db)


# ── CLASIFICAR + GUARDAR SESIÓN ────────────────────
@app.post("/clasificar")
def clasificar_sesion(
    data: ClasificarRequest,
    token: str,
    db: Session = Depends(get_db)
):
    user = get_current_user(token, db)

    # Convertir métricas a lista de dicts
    metricas_dict = [m.dict() for m in data.metricas]

    # Clasificar con XGBoost
    resultado = clasificar(metricas_dict)

    # Guardar sesión en BD
    sesion = models.Sesion(
        usuario_id       = user.id,
        duracion         = data.duracion,
        estilo_cognitivo = resultado["estilo"],
        confianza        = resultado["confianza"],
    )
    db.add(sesion)
    db.commit()
    db.refresh(sesion)

    # Guardar métricas individuales
    for m in metricas_dict:
        gaze = models.GazeDato(
            sesion_id   = sesion.id,
            timestamp   = int(datetime.utcnow().timestamp() * 1000),
            yaw         = m.get("yaw"),
            pitch       = m.get("pitch"),
            roll        = m.get("roll"),
            blink_ratio = m.get("blink_ratio"),
            gaze_x      = m.get("gaze_x"),
            gaze_y      = m.get("gaze_y"),
            pupil_px    = m.get("pupil_px"),
        )
        db.add(gaze)
    db.commit()

    return {
        "estilo_cognitivo": resultado["estilo"],
        "confianza":        resultado["confianza"],
        "sesion_id":        sesion.id,
        "mensaje":          f"Clasificado como {resultado['estilo']} con {round(resultado['confianza']*100)}% de confianza"
    }


# Historial del usuario
@app.get("/sesiones/me", response_model=List[schemas.SesionOut])
def get_mis_sesiones(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    return db.query(models.Sesion).filter(
        models.Sesion.usuario_id == user.id
    ).order_by(models.Sesion.fecha.desc()).all()

# Actualizar perfil
@app.put("/usuarios/me/update")
def update_me(
    data: dict,
    token: str,
    db: Session = Depends(get_db)
):
    user = get_current_user(token, db)
    if "nombre"   in data: user.nombre  = data["nombre"]
    if "carrera"  in data: user.carrera = data["carrera"]
    if "password" in data and data["password"]:
        user.password = hash_password(data["password"])
    db.commit()
    db.refresh(user)
    return {"mensaje": "Perfil actualizado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)