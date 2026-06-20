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

import json

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

def hash_password(p):      return pwd_context.hash(p)
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
    return {"access_token": token, "token_type": "bearer", "rol": user.rol, "nombre": user.nombre, "id": user.id}


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

    metricas_dict = [m.dict() for m in data.metricas]
    resultado     = clasificar(metricas_dict)

    sesion = models.Sesion(
        usuario_id       = user.id,
        duracion         = data.duracion,
        estilo_cognitivo = resultado["estilo"],
        confianza        = resultado["confianza"],
    )
    db.add(sesion)
    db.commit()
    db.refresh(sesion)

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
def update_me(data: dict, token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    if "nombre"   in data: user.nombre  = data["nombre"]
    if "carrera"  in data: user.carrera = data["carrera"]
    if "password" in data and data["password"]:
        user.password = hash_password(data["password"])
    db.commit()
    db.refresh(user)
    return {"mensaje": "Perfil actualizado"}


# ── ADMIN ──────────────────────────────────────────

def require_admin(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    if user.rol != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores")
    return user

@app.get("/admin/usuarios", response_model=List[schemas.UsuarioOut])
def admin_get_usuarios(token: str, db: Session = Depends(get_db)):
    require_admin(token, db)
    return db.query(models.Usuario).all()

@app.delete("/admin/usuarios/{user_id}", status_code=204)
def admin_delete_usuario(user_id: int, token: str, db: Session = Depends(get_db)):
    require_admin(token, db)
    user = db.query(models.Usuario).filter(models.Usuario.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    db.delete(user)
    db.commit()

@app.put("/admin/usuarios/{user_id}", response_model=schemas.UsuarioOut)
def admin_update_usuario(user_id: int, data: schemas.UsuarioUpdate, token: str, db: Session = Depends(get_db)):
    require_admin(token, db)
    user = db.query(models.Usuario).filter(models.Usuario.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    for field, value in data.dict(exclude_unset=True).items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user

@app.get("/admin/sesiones/total")
def admin_get_total_sesiones(token: str, db: Session = Depends(get_db)):
    require_admin(token, db)
    return {"total": db.query(models.Sesion).count()}

@app.get("/admin/usuarios/{user_id}/sesiones")
def admin_get_sesiones_usuario(user_id: int, token: str, db: Session = Depends(get_db)):
    require_admin(token, db)
    sesiones = db.query(models.Sesion).filter(
        models.Sesion.usuario_id == user_id
    ).order_by(models.Sesion.fecha.desc()).all()
    return sesiones

@app.get("/admin/usuarios/{user_id}/cuestionario")
def admin_get_cuestionario_usuario(user_id: int, token: str, db: Session = Depends(get_db)):
    require_admin(token, db)
    cuestionario = db.query(models.Cuestionario).filter(
        models.Cuestionario.usuario_id == user_id
    ).order_by(models.Cuestionario.fecha.desc()).first()
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Sin cuestionario")
    return cuestionario

@app.get("/admin/usuarios/{user_id}/cuestionarios", response_model=List[schemas.CuestionarioOut])
def admin_get_cuestionarios_usuario(user_id: int, token: str, db: Session = Depends(get_db)):
    require_admin(token, db)
    return db.query(models.Cuestionario).filter(
        models.Cuestionario.usuario_id == user_id
    ).order_by(models.Cuestionario.fecha.desc()).all()

# ── CUESTIONARIO FELDER-SILVERMAN ──────────────────

def calcular_resultado_cuestionario(respuestas: dict) -> dict:
    """
    Calcula el puntaje Visual/Verbal.
    a = Visual (+1), b = Verbal (-1)
    Puntaje: -11 a +11
    """
    puntaje = 0
    for v in respuestas.values():
        if v == 'a':
            puntaje += 1
        elif v == 'b':
            puntaje -= 1

    if puntaje >= 4:
        resultado = "Visual"
    elif puntaje <= -4:
        resultado = "Verbal"
    else:
        resultado = "Balanceado"

    return {"puntaje": puntaje, "resultado": resultado}


@app.post("/cuestionario", response_model=schemas.CuestionarioOut, status_code=201)
def crear_cuestionario(
    data: schemas.CuestionarioCreate,
    token: str,
    db: Session = Depends(get_db)
):
    user = get_current_user(token, db)
    calc = calcular_resultado_cuestionario(data.respuestas)

    cuestionario = models.Cuestionario(
        usuario_id = user.id,
        respuestas = json.dumps(data.respuestas),
        puntaje    = calc["puntaje"],
        resultado  = calc["resultado"],
    )
    db.add(cuestionario)
    db.commit()
    db.refresh(cuestionario)
    return cuestionario


@app.get("/cuestionario/me", response_model=schemas.CuestionarioOut)
def get_mi_cuestionario(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    cuestionario = db.query(models.Cuestionario).filter(
        models.Cuestionario.usuario_id == user.id
    ).order_by(models.Cuestionario.fecha.desc()).first()
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario no completado aún.")
    return cuestionario


@app.get("/cuestionario/historial", response_model=List[schemas.CuestionarioOut])
def get_historial_cuestionarios(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    return db.query(models.Cuestionario).filter(
        models.Cuestionario.usuario_id == user.id
    ).order_by(models.Cuestionario.fecha.asc()).all()

# ── CUESTIONARIO OSIVQ ─────────────────────────────

# Ítems que se invierten (carga negativa en su factor)
ITEMS_INVERTIDOS_OBJECT = {10, 25}
ITEMS_INVERTIDOS_VERBAL = {24, 38}

# Ítems por dimensión
ITEMS_OBJECT = {6, 11, 20, 23, 26, 29, 33, 34, 40, 43, 45, 18, 13, 10, 25}
ITEMS_VERBAL = {2, 4, 8, 9, 16, 21, 35, 36, 37, 39, 41, 28, 19, 24, 38}

def calcular_resultado_osivq(respuestas: dict) -> dict:
    """
    Calcula puntajes Object y Verbal del OSIVQ.
    Escala Likert 1-5. Ítems invertidos: 6 - valor.
    Umbral de empate: diferencia <= 5 → Balanceado.
    """
    puntaje_object = 0
    puntaje_verbal = 0

    for item_str, valor in respuestas.items():
        item = int(item_str)
        v = int(valor)

        if item in ITEMS_OBJECT:
            if item in ITEMS_INVERTIDOS_OBJECT:
                v = 6 - v
            puntaje_object += v

        elif item in ITEMS_VERBAL:
            if item in ITEMS_INVERTIDOS_VERBAL:
                v = 6 - v
            puntaje_verbal += v

    diferencia = abs(puntaje_object - puntaje_verbal)

    if diferencia <= 5:
        resultado = "Balanceado"
    elif puntaje_object > puntaje_verbal:
        resultado = "Visual"
    else:
        resultado = "Verbal"

    return {
        "puntaje_object": puntaje_object,
        "puntaje_verbal": puntaje_verbal,
        "resultado":      resultado,
    }


@app.post("/cuestionario-osivq", response_model=schemas.CuestionarioOSIVQOut, status_code=201)
def crear_cuestionario_osivq(
    data: schemas.CuestionarioOSIVQCreate,
    token: str,
    db: Session = Depends(get_db)
):
    user = get_current_user(token, db)
    calc = calcular_resultado_osivq(data.respuestas)

    cuestionario = models.CuestionarioOSIVQ(
        usuario_id     = user.id,
        respuestas     = json.dumps(data.respuestas),
        puntaje_object = calc["puntaje_object"],
        puntaje_verbal = calc["puntaje_verbal"],
        resultado      = calc["resultado"],
    )
    db.add(cuestionario)
    db.commit()
    db.refresh(cuestionario)
    return cuestionario


@app.get("/cuestionario-osivq/me", response_model=schemas.CuestionarioOSIVQOut)
def get_mi_cuestionario_osivq(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    cuestionario = db.query(models.CuestionarioOSIVQ).filter(
        models.CuestionarioOSIVQ.usuario_id == user.id
    ).order_by(models.CuestionarioOSIVQ.fecha.desc()).first()
    if not cuestionario:
        raise HTTPException(status_code=404, detail="Cuestionario OSIVQ no completado aún.")
    return cuestionario


@app.get("/cuestionario-osivq/historial", response_model=List[schemas.CuestionarioOSIVQOut])
def get_historial_osivq(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    return db.query(models.CuestionarioOSIVQ).filter(
        models.CuestionarioOSIVQ.usuario_id == user.id
    ).order_by(models.CuestionarioOSIVQ.fecha.asc()).all()


@app.get("/admin/usuarios/{user_id}/cuestionarios-osivq", response_model=List[schemas.CuestionarioOSIVQOut])
def admin_get_cuestionarios_osivq_usuario(user_id: int, token: str, db: Session = Depends(get_db)):
    require_admin(token, db)
    return db.query(models.CuestionarioOSIVQ).filter(
        models.CuestionarioOSIVQ.usuario_id == user_id
    ).order_by(models.CuestionarioOSIVQ.fecha.desc()).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)