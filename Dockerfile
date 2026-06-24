FROM python:3.12-slim

# Instalar uv desde su imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Instalar dependencias primero para aprovechar el cache de capas
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copiar el código fuente
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
