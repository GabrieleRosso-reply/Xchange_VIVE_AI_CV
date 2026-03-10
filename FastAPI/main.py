def analyze_led_circuit(img):
    if img is None:
        return {"status": "errore"}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    blurred_v = cv2.GaussianBlur(v_channel, (15, 15), 0)

    _, max_val, _, max_loc = cv2.minMaxLoc(blurred_v)
    x, y = max_loc

    if max_val < 150:
        return {"state": "Spento", "confidence": 0.0}

    padding = 40
    h, w = img.shape[:2]

    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(w, x + padding), min(h, y + padding)

    led_crop = img[y1:y2, x1:x2]
    hsv_crop = hsv[y1:y2, x1:x2]

    color_mask = cv2.inRange(hsv_crop, (0, 60, 60), (180, 255, 255))
    valid_pixels = hsv_crop[color_mask > 0]

    if len(valid_pixels) == 0:
        return {"state": "Indeterminato", "confidence": 0.0}

    color_ranges = {
        "Rosso": [(0, 10), (165, 180)],
        "Giallo": [(15, 35)],
        "Verde": [(45, 85)],
        "Blu": [(100, 130)],
        "Viola": [(135, 160)]
    }

    hues = valid_pixels[:, 0]
    votes = {}

    for color, ranges in color_ranges.items():
        count = sum(np.sum((hues >= low) & (hues <= high)) for low, high in ranges)
        votes[color] = count

    winner = max(votes, key=votes.get)
    confidence = votes[winner] / len(valid_pixels)

    return {
        "state": "Acceso",
        "color": winner,
        "confidence": float(confidence)
    }

from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI()

@app.post("/analyze-led")
async def analyze_led(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = analyze_led_circuit(img)

    return result