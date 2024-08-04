from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from api import controller

app = FastAPI(docs_url='/docs/api')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(controller.router)

router = APIRouter(
    tags = ["Chat API endpoints"],
    prefix = "/api"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)