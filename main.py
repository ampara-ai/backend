import os
import uvicorn

if __name__ == "__main__":
    reload = os.getenv("APP_ENV", "development") == "development"
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=reload)
