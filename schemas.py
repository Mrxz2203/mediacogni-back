from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# ── Auth ──────────────────────────────────────────
class RegisterSchema(BaseModel):
    nombre   : str
    codigo   : str
    password : str
    rol      : str        # "estudiante" | "docente"
    carrera  : Optional[str] = None

class LoginSchema(BaseModel):
    codigo   : str
    password : str

class TokenSchema(BaseModel):
    access_token : str
    token_type   : str = "bearer"
    rol          : str
    nombre       : str
    id           : int  

# ── Sesiones ──────────────────────────────────────
class SesionCreate(BaseModel):
    duracion         : int   = 90
    estilo_cognitivo : Optional[str]   = None
    confianza        : Optional[float] = None

class SesionOut(BaseModel):
    id               : int
    fecha            : datetime
    duracion         : int
    estilo_cognitivo : Optional[str]
    confianza        : Optional[float]

    class Config:
        from_attributes = True

# ── Usuario ───────────────────────────────────────
class UsuarioOut(BaseModel):
    id      : int
    nombre  : str
    codigo  : str
    rol     : str
    carrera : Optional[str]
    class Config:
        from_attributes = True

# ── Admin ───────────────────────────────────────
class UsuarioUpdate(BaseModel):
    nombre  : Optional[str] = None
    codigo  : Optional[str] = None
    rol     : Optional[str] = None
    carrera : Optional[str] = None
    
    

class CuestionarioCreate(BaseModel):
    respuestas: dict  # {"1": "a", "2": "b", ...}

class CuestionarioOut(BaseModel):
    id:        int
    puntaje:   int
    resultado: str
    fecha:     datetime
    respuestas: str

    class Config:
        from_attributes = True
        

class CuestionarioOSIVQCreate(BaseModel):
    respuestas: dict  # {"1": 3, "2": 5, ...} valores del 1 al 5

class CuestionarioOSIVQOut(BaseModel):
    id:             int
    puntaje_object: int
    puntaje_verbal: int
    resultado:      str
    fecha:          datetime
    respuestas:     str

    class Config:
        from_attributes = True
