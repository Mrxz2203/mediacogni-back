from sqlalchemy import Column, Integer, BigInteger, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import enum

class RolEnum(str, enum.Enum):
    estudiante = "estudiante"
    docente    = "docente"
    admin      = "admin"

class Usuario(Base):
    __tablename__ = "usuarios"

    id       = Column(Integer, primary_key=True, index=True)
    nombre   = Column(String, nullable=False)
    codigo   = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    rol      = Column(Enum(RolEnum), nullable=False)
    carrera  = Column(String, nullable=True)

    sesiones      = relationship("Sesion", back_populates="usuario")
    cuestionarios = relationship("Cuestionario", back_populates="usuario")


class Sesion(Base):
    __tablename__ = "sesiones"

    id               = Column(Integer, primary_key=True, index=True)
    usuario_id       = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    fecha            = Column(DateTime(timezone=True), server_default=func.now())
    duracion         = Column(Integer, default=90)
    estilo_cognitivo = Column(String, nullable=True)
    confianza        = Column(Float, nullable=True)

    usuario   = relationship("Usuario", back_populates="sesiones")
    gaze_data = relationship("GazeDato", back_populates="sesion")


class GazeDato(Base):
    __tablename__ = "gaze_datos"

    id          = Column(Integer, primary_key=True, index=True)
    sesion_id   = Column(Integer, ForeignKey("sesiones.id"), nullable=False)
    timestamp   = Column(BigInteger)
    yaw         = Column(Float)
    pitch       = Column(Float)
    roll        = Column(Float)
    blink_ratio = Column(Float)
    gaze_x      = Column(Float)
    gaze_y      = Column(Float)
    pupil_px    = Column(Float)

    sesion = relationship("Sesion", back_populates="gaze_data")


class Cuestionario(Base):
    __tablename__ = "cuestionarios"

    id         = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    respuestas = Column(String, nullable=False)
    puntaje    = Column(Integer, nullable=False)
    resultado  = Column(String, nullable=False)
    fecha      = Column(DateTime(timezone=True), server_default=func.now())

    usuario = relationship("Usuario", back_populates="cuestionarios") 