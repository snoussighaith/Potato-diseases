

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Potato Disease Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "saved_models" / "mobilenetv2_augmented.keras"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = (224, 224)
RECOMMENDATIONS = {
    "Early Blight": "Retirez les feuilles atteintes et appliquez un traitement fongicide adapté.",
    "Late Blight": "Isolez les plants atteints et appliquez rapidement un traitement anti-mildiou.",
    "Healthy": "La plante semble saine ; continuez la surveillance et les bonnes pratiques culturales.",
}

MODEL: Optional[tf.keras.Model] = None
MODEL_LOAD_ERROR: Optional[str] = None

if MODEL_PATH.exists():
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except (OSError, ValueError) as exc:
        MODEL_LOAD_ERROR = f"Impossible de charger le modèle {MODEL_PATH}: {exc}"
else:
    MODEL_LOAD_ERROR = f"Modèle introuvable : {MODEL_PATH}"


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}


def prepare_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as source_image:
            image = source_image.convert("RGB").resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Le fichier fourni n'est pas une image valide.",
        ) from exc

    image_array = np.asarray(image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    return preprocess_input(image_batch)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=MODEL_LOAD_ERROR or "Le modèle n'est pas disponible.",
        )

    image_data = await file.read()

    if not image_data:
        raise HTTPException(status_code=400, detail="Le fichier est vide.")

    image_batch = prepare_image(image_data)
    try:
        predictions = MODEL.predict(image_batch, verbose=0)
    except (TypeError, ValueError, tf.errors.OpError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Erreur pendant la prédiction du modèle.",
        ) from exc

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]

    return {
        "class": predicted_class,
        "confidence": confidence,
        "recommendation": RECOMMENDATIONS[predicted_class],
    }