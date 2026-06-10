# create_admin.py
from database import SessionLocal
from models import Usuario
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

db = SessionLocal()

admin = Usuario(
    nombre   = "Administrador",
    codigo   = "admin001",        # el "usuario" con el que se loguea
    password = pwd_context.hash("admin1234"),   # contraseña hasheada
    rol      = "admin",
    carrera  = None
)

db.add(admin)
db.commit()
db.close()
print("Admin creado.")
