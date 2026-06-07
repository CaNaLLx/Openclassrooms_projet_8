from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from predict import predict_mask

app = FastAPI(title="Segmentation API")

@app.get("/")
def root():
    return {"message": "API de segmentation - Future Vision Transport"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    mask_bytes = predict_mask(image_bytes)
    return Response(content=mask_bytes, media_type="image/png")
