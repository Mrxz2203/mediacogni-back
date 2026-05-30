"""
api.py — FastAPI para V-COGNI
Ejecutar con: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List
from jose import JWTError, jwt
from passlib.context import CryptContext

import models, schemas
from database import engine, get_db

# ── Crear tablas automáticamente ──────────────────
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="V-COGNI API", version="1.0")

# ── CORS — permite conexión desde React ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── JWT config ────────────────────────────────────
SECRET_KEY  = "vcogni_secret_key_2026"
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        codigo: str = payload.get("sub")
        if not codigo:
            raise HTTPException(status_code=401, detail="Token inválido")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")
    user = db.query(models.Usuario).filter(models.Usuario.codigo == codigo).first()
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    return user


# ── ENDPOINTS ─────────────────────────────────────

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


# Guardar sesión
@app.post("/sesiones", response_model=schemas.SesionOut, status_code=201)
def create_sesion(data: schemas.SesionCreate, token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    sesion = models.Sesion(
        usuario_id       = user.id,
        duracion         = data.duracion,
        estilo_cognitivo = data.estilo_cognitivo,
        confianza        = data.confianza,
    )
    db.add(sesion)
    db.commit()
    db.refresh(sesion)
    return sesion


# Historial del usuario
@app.get("/sesiones/me", response_model=List[schemas.SesionOut])
def get_mis_sesiones(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    return db.query(models.Sesion).filter(
        models.Sesion.usuario_id == user.id
    ).order_by(models.Sesion.fecha.desc()).all()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)