from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import enum

class RolEnum(str, enum.Enum):
    estudiante = "estudiante"
    docente = "docente"

class Usuario(Base):
    __tablename__ = "usuarios"

    id       = Column(Integer, primary_key=True, index=True)
    nombre   = Column(String, nullable=False)
    codigo   = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    rol      = Column(Enum(RolEnum), nullable=False)
    carrera  = Column(String, nullable=True)

    sesiones = relationship("Sesion", back_populates="usuario")


class Sesion(Base):
    __tablename__ = "sesiones"

    id               = Column(Integer, primary_key=True, index=True)
    usuario_id       = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    fecha            = Column(DateTime(timezone=True), server_default=func.now())
    duracion         = Column(Integer, default=90)       # segundos
    estilo_cognitivo = Column(String, nullable=True)     # "Visual" | "Verbal"
    confianza        = Column(Float, nullable=True)       # 0.0 - 1.0

    usuario   = relationship("Usuario", back_populates="sesiones")
    gaze_data = relationship("GazeDato", back_populates="sesion")


class GazeDato(Base):
    __tablename__ = "gaze_datos"

    id          = Column(Integer, primary_key=True, index=True)
    sesion_id   = Column(Integer, ForeignKey("sesiones.id"), nullable=False)
    timestamp   = Column(Integer)
    yaw         = Column(Float)
    pitch       = Column(Float)
    roll        = Column(Float)
    blink_ratio = Column(Float)
    gaze_x      = Column(Float)
    gaze_y      = Column(Float)
    pupil_px    = Column(Float)

    sesion = relationship("Sesion", back_populates="gaze_data")