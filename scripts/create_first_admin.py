"""
Crea el primer usuario administrador en la base de datos PostgreSQL.

Uso:
    uv run python scripts/create_first_admin.py --username admin --password mysecretpassword

Solo funciona si no existe ningún admin en la base de datos (protección contra
sobrescribir credenciales existentes). Para crear admins adicionales, usar el
endpoint POST /api/v1/admin/users con un token válido.
"""

import argparse
import asyncio
import sys

# Allow running from project root
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.services import postgres
from app.services.auth import hash_password


async def main(username: str, password: str) -> None:
    if not settings.postgres_dsn:
        print("ERROR: POSTGRES_DSN no está configurado en .env")
        sys.exit(1)

    await postgres.create_tables_if_not_exist()

    count = await postgres.count_admins()
    if count > 0:
        print(
            f"ERROR: Ya existe(n) {count} admin(s) en la base de datos.\n"
            "Para crear más admins usa el endpoint POST /api/v1/admin/users con tu token JWT."
        )
        sys.exit(1)

    pw_hash = hash_password(password)
    row = await postgres.create_admin(username, pw_hash)
    print(f"Admin creado exitosamente:")
    print(f"  id       : {row['id']}")
    print(f"  username : {row['username']}")
    print(f"  created  : {row['created_at']}")
    print("\nAhora puedes hacer login en POST /api/v1/admin/login")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crear el primer usuario admin de Ampara")
    parser.add_argument("--username", required=True, help="Nombre de usuario del admin")
    parser.add_argument("--password", required=True, help="Contraseña del admin")
    args = parser.parse_args()

    asyncio.run(main(args.username, args.password))
