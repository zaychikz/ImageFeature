import base64
from fastapi import FastAPI,Request
import cv2
import numpy as np
from app.hog import gethog

app = FastAPI()

def read64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

@app.get("/")
def root():
    return {"message": "This is my api"}

@app.get("/api/gethog")
async def read_str(request : Request):
    item = await request.json()
    item_str = item['img']
    img = read64(item_str)
    hog = gethog(img)
    return {"hog":hog.tolist()}