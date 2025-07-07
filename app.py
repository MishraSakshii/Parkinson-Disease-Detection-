# app.py (using state_dict loading - Option 2)
from flask import Flask, request, jsonify, render_template, send_file
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import io
import os
from dotenv import load_dotenv
from fpdf import FPDF
from datetime import datetime
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

app = Flask(__name__)

# Define and load model using state_dict
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device("cpu")))
model.eval()

# Globals to store prediction info
latest_prediction = None
confidence_last = 0.0
latest_ai_response = ""
uploaded_filename = ""
user_name = ""

# Image transform
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# Predict
def predict(image_bytes):
    global latest_prediction, confidence_last
    tensor = transform_image(image_bytes)
    output = model(tensor)
    prediction = torch.argmax(output, dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][prediction].item()
    latest_prediction = "Static" if prediction == 0 else "Dynamic"
    confidence_last = round(confidence * 100, 2)
    return latest_prediction, confidence_last

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    global uploaded_filename
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['image']
    image_bytes = file.read()
    os.makedirs("uploads", exist_ok=True)
    uploaded_filename = os.path.join("uploads", file.filename)
    with open(uploaded_filename, "wb") as f:
        f.write(image_bytes)
    result, confidence = predict(image_bytes)
    return jsonify({'result': result, 'accuracy': confidence})

@app.route('/chat', methods=['POST'])
def chat():
    global latest_ai_response
    data = request.get_json()
    user_message = data.get("message", "")
    context = f"The user's handwriting image was predicted as '{latest_prediction}'." if latest_prediction else ""
    try:
        response = openai.ChatCompletion.create(
            model="mistralai/mixtral-8x7b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful AI advisor for Parkinson’s patients. " + context},
                {"role": "user", "content": user_message}
            ]
        )
        reply = response['choices'][0]['message']['content']
        latest_ai_response = reply
        return jsonify({'response': reply})
    except Exception as e:
        print("OpenRouter error:", e)
        return jsonify({'response': f"Error: {str(e)}"})

@app.route('/set-name', methods=['POST'])
def set_name():
    global user_name
    data = request.get_json()
    user_name = data.get("name", "")
    return jsonify({"message": "Name saved."})

@app.route('/download-report')
def download_report():
    if not latest_prediction or not uploaded_filename:
        return jsonify({'error': 'No prediction yet.'}), 400
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Parkinson's Diagnosis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Name: {user_name}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {latest_prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence_last}%", ln=True)
    pdf.ln(10)

    if uploaded_filename and os.path.exists(uploaded_filename):
        y_before_image = pdf.get_y()
        pdf.image(uploaded_filename, x=10, y=y_before_image, w=100)
        pdf.set_y(y_before_image + 85)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="AI Advisor's Suggestions:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=latest_ai_response)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt="Note: This report is generated using AI-based analysis and is not a substitute for a medical diagnosis. Always consult a healthcare professional.")

    os.makedirs("reports", exist_ok=True)
    filename = f"parkinson_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("reports", filename)
    pdf.output(filepath)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
