from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os

app = FastAPI()

API_KEY = os.getenv("CV_API_KEY", "supersegreta")

def analyze_led_circuit_from_bytes(image_bytes: bytes):
 
    # Decodifica bytes -> immagine OpenCV (BGR)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Impossibile decodificare l'immagine")

    # Conversione in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # Blur per ridurre il rumore
    blurred_v = cv2.GaussianBlur(v_channel, (15, 15), 0)

    # Punto più luminoso
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred_v)
    x, y = max_loc

    # Se troppo scuro -> LED spento / non rilevato
    if max_val < 150:
        return {
            "category": "circuito",
            "value": "spento",
            "confidence": 0.0
        }

    # Crop attorno al punto più luminoso
    padding = 40
    h, w = img.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(w, x + padding), min(h, y + padding)

    hsv_crop = hsv[y1:y2, x1:x2]

    # Maschera per escludere centro bianco e pixel scuri
    color_mask = cv2.inRange(hsv_crop, (0, 60, 60), (180, 255, 255))
    valid_pixels = hsv_crop[color_mask > 0]

    if len(valid_pixels) == 0:
        return {
            "category": "circuito",
            "value": "indeterminato",
            "confidence": 0.0
        }

    # Range Hue
    color_ranges = {
        "rosso": [(0, 10), (165, 180)],
        "giallo": [(15, 35)],
        "verde": [(45, 85)],
        "blu": [(100, 130)],
        "viola": [(135, 160)]
    }

    hues = valid_pixels[:, 0]
    votes = {}

    for color, ranges in color_ranges.items():
        count = sum(np.sum((hues >= low) & (hues <= high)) for low, high in ranges)
        votes[color] = int(count)

    winner = max(votes, key=votes.get)
    confidence = votes[winner] / len(valid_pixels)

    # Soglia minima consigliata per evitare classificazioni forzate
    if confidence < 0.20:
        return {
            "category": "circuito",
            "value": "indeterminato",
            "confidence": round(float(confidence), 4)
        }

    return {
        "category": "circuito",
        "value": winner,
        "confidence": round(float(confidence), 4)
    }


@app.get("/")
def root():
    return {"message": "LED Color Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
   # Auth semplice con API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # Verifica content-type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image_bytes = await file.read()
        result = analyze_led_circuit_from_bytes(image_bytes)

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
