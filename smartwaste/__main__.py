"""Allow running the web UI with: python -m smartwaste"""

from .web import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    from .config import WEB_HOST, WEB_PORT

    uvicorn.run("smartwaste.web:app", host=WEB_HOST, port=WEB_PORT)
