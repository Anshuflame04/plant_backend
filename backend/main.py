import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import json
import os

# Load models and disease details
model_directory = 'backend/model'
data_directory = 'backend/data'

models = {
    'tomato': tf.keras.models.load_model(os.path.join(model_directory, 'Tomato_model_v4.keras')),
    'potato': tf.keras.models.load_model(os.path.join(model_directory, 'Potato_model_v2.keras')),
    'rice': tf.keras.models.load_model(os.path.join(model_directory, 'Rice_model_v3.keras')),
    'wheat': tf.keras.models.load_model(os.path.join(model_directory, 'Rice_model_v3.keras')),  # Make sure you use the correct model for wheat
    'apple': tf.keras.models.load_model(os.path.join(model_directory, 'apple_model_v1.keras'))
}

disease_details = {
    'tomato': json.load(open(os.path.join(data_directory, 'TomatoDetails.json'))),
    'potato': json.load(open(os.path.join(data_directory, 'PotatoDetails.json'))),
    'rice': json.load(open(os.path.join(data_directory, 'RiceDetails.json'))),
    'wheat': json.load(open(os.path.join(data_directory, 'RiceDetails.json'))),  # Ensure you have the correct details for wheat
    'apple': json.load(open(os.path.join(data_directory, 'AppleDetails.json')))
}

# Create FastAPI application
app = FastAPI()

# Set up CORS
origins = ["http://localhost:3000", "https://your-react-app.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Update this list with your production URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocess the input image
def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize image to match model input size
    image_array = np.array(image)
    image_array = preprocess_input(image_array)  # Preprocess the image for MobileNet
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Predict endpoint
@app.post("/predict/{crop_name}")
async def predict(crop_name: str, file: UploadFile = File(...)):
    if crop_name not in models:
        return JSONResponse(content={"error": "Invalid crop name."}, status_code=400)

    try:
        model = models[crop_name]  # Load the appropriate model
        disease_dict = disease_details[crop_name]  # Load disease details

        image = Image.open(file.file)  # Open the uploaded image
        preprocessed_image = preprocess_image(image)  # Preprocess the image

        # Make predictions
        predictions = model.predict(preprocessed_image)
        class_index = np.argmax(predictions)  # Get index of highest prediction
        confidence = np.max(predictions[0]) * 100  # Get confidence score

        class_names = list(disease_dict.keys())
        class_name = class_names[class_index]  # Get the predicted class name

        disease_info = disease_dict.get(class_name, {
            'name': 'Unknown',
            'cause': 'Unknown cause.',
            'prevention': 'No prevention available.',
            'medicines': 'No medicines available.'
        })

        return JSONResponse(content={
            "predicted_disease": disease_info.get('name'),
            "confidence_score": confidence,
            "cause": disease_info.get('cause'),
            "prevention": disease_info.get('prevention'),
            "medicines": disease_info.get('medicines'),
        }, status_code=200)

    except Exception as e:
        return JSONResponse(content={
            "error": f"An error occurred during prediction: {str(e)}"
        }, status_code=500)

