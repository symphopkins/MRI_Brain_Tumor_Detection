from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from io import BytesIO

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model_path = "/Users/symphonyhopkins/Documents/Data Projects/Github/MRI_Model/model.h5"
model = tf.keras.models.load_model(model_path)

# Load label encoder
label_encoder_path = "/Users/symphonyhopkins/Documents/Data Projects/Github/MRI_Model/label_encoder.npy"
label_encoder = np.load(label_encoder_path, allow_pickle=True)

labels = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

# Define preprocess image function
def preprocess_image(file):
    img = image.load_img(BytesIO(file.read()), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
async def index():
    try:
        with open("index.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("Received image file:", file.filename)

        # Preprocess image
        img_array = preprocess_image(file.file)

        print("Image preprocessed successfully.")

        # Predict the class probabilities
        preds = model.predict(img_array)

        print("Prediction completed.")

        # Map the predicted class index to label name
        predicted_label = labels[np.argmax(preds)]

        print("Predicted label:", predicted_label) 

        return JSONResponse(content={"prediction": predicted_label})
    except Exception as e:
        print("Error occurred:", e) 
        return JSONResponse(content={"error": str(e)})
