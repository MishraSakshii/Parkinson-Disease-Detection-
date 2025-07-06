from flask import Flask, request, jsonify, render_template
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import io
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device("cpu")))
model.eval()

# Global variable to track last prediction
latest_prediction = None

# Image preprocessing function
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# Predict Parkinson’s class
def predict(image_bytes):
    global latest_prediction
    tensor = transform_image(image_bytes)
    output = model(tensor)
    prediction = torch.argmax(output, dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][prediction].item()
    latest_prediction = "Static" if prediction == 0 else "Dynamic"
    return latest_prediction, round(confidence * 100, 2)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Image classification route
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    image = request.files['image'].read()
    result, confidence = predict(image)
    return jsonify({'result': result, 'accuracy': confidence})

# AI chat advisor using OpenRouter
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    context = f"The user's handwriting image was predicted as '{latest_prediction}'." if latest_prediction else ""

    try:
        response = openai.ChatCompletion.create(
            model="mistralai/mixtral-8x7b-instruct",  # You can change to openai/gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful AI advisor for Parkinson’s patients. " + context},
                {"role": "user", "content": user_message}
            ]
        )
        reply = response['choices'][0]['message']['content']
        return jsonify({'response': reply})
    except Exception as e:
        print("OpenRouter error:", e)
        return jsonify({'response': f"Error: {str(e)}"})

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
