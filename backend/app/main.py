import sys
from pathlib import Path

# sys.path bridge to project root for "import src.*"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from app.schema import schema
from app.services.model_loader import ModelStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once at startup
    ModelStore.load()
    yield
    # Cleanup if needed

app = FastAPI(
    title="OSFDA Aviation Safety API",
    description="GraphQL API for Aviation Safety Feature Discovery and Analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount GraphQL
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": ModelStore.ready,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
