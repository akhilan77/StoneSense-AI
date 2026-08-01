from fastapi import FastAPI

app = FastAPI(
    title="StoneSense AI API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "StoneSense AI Backend Running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}
