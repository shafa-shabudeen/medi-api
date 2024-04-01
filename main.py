from fastapi import FastAPI , File , UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import io
from fastapi.responses import JSONResponse

# Import statements

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TensorFlow model outside the application initialization
model = tf.keras.models.load_model("tf_trained_model.h5")
class_names = [...]
...

# Define the reshape_image function
...

@app.get("/ping")
async def ping():
    return "HELLO"


@app.post("/predict")
async def Predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = reshape_image(contents)
    prediction_array = model.predict(img_array)
    predicted_label = np.argmax(prediction_array)
    predicted_class = class_names[predicted_label]
    return JSONResponse(content=predicted_class)


# Use environment variables for configuration
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
