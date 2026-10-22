"""Product similarity search for the SAP CX technical exercise."""


def __getattr__(name: str):
    # Lets the provided Dockerfile's `uvicorn app:app` find the FastAPI instance without eager imports.
    if name == "app":
        from app.api import app

        return app
    raise AttributeError(name)
