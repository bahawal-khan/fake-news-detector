from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .predict import get_model
from .preprocess import has_sufficient_text_content
from .schemas import HealthResponse, PredictRequest, PredictResponse

app = FastAPI(
    title="Fake News Detector API",
    description="ML-based (TF-IDF + handcrafted features + LinearSVC) fake news classifier.",
    version="1.0.0",
)

# Allow the React frontend (any origin in dev; lock this down in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup():
    # Warms up the model once so the first real request isn't slow
    get_model()


@app.get("/", tags=["Health"])
def root():
    return {"message": "Fake News Detector API is running. See /docs for usage."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    try:
        model = get_model()
        return HealthResponse(status="ok", model_loaded=True, model_name=model.model_name)
    except Exception:
        return HealthResponse(status="error", model_loaded=False, model_name="none")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(payload: PredictRequest):
    if not payload.text or len(payload.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Article text is too short to analyze.")

    if not has_sufficient_text_content(payload.text):
        raise HTTPException(
            status_code=400,
            detail=(
                "This doesn't look like readable article text (too few real words "
                "after removing numbers/symbols). Please paste actual sentences."
            ),
        )

    model = get_model()
    result = model.predict(payload.title, payload.text)
    return PredictResponse(**result)
