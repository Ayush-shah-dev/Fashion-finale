from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analyze

app = FastAPI(title="Fashionlytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)

@app.get("/health")
def health():
    return {"status": "ok"}
