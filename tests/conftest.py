import os

# app.core.config exige estas variables al importarse; los tests no llaman a
# ningún servicio real.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
