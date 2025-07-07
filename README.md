# Parkinson's Disease Detection + AI Health Advisor

A deep learning-powered web application that analyzes handwriting samples to assist in Parkinson's disease screening. It integrates image-based prediction, an AI health advisor (via OpenRouter or Gemini), and automatic PDF report generation.

---

##  Features

-  **Handwriting Pattern Classification** (Static vs Dynamic)
-  **AI Health Advisor Chat** (OpenRouter or Gemini-backed)
-  **Auto-generated PDF Report** (with uploaded image + advice)
-  **ResNet18-based Trained Model**
-  **Web UI Built in Flask**
-  **Trainable & Extendable Model Pipeline**

---

## Model Information

- Architecture: `ResNet18` (transfer learning)
- Classes: `Static` (rigid strokes), `Dynamic` (fluid strokes)
- Input size: 224×224 RGB images
- Augmented with rotations, flips

### Prediction Output:
- Label: `Static` or `Dynamic`
- Confidence: softmax score (%)

---

## Folder Structure

```
parkinson_advisor_project/
├── app.py                   # Flask app for frontend & backend logic
├── train_model.py          # Script to train and save model weights
├── model_weights.pth       # Trained model (state_dict)
├── static/
│   └── background.jpg      # Background image for web UI
├── templates/
│   └── index.html          # HTML template (UI with image + chat)
├── dataset/
│   ├── Static/             # Class 0 handwriting images
│   └── Dynamic/            # Class 1 handwriting images
```

---

## Pretrained Models

|         Type         |                                      Download Link                                             |
|----------------------|------------------------------------------------------------------------------------------------|
| `model_weights.pth`  | [Download](https://drive.google.com/file/d/1kHkBotIRosTOy1EaxUYuRvJse081Gf3s/view?usp=sharing) |
| `model.pth` (legacy) | [Download](https://drive.google.com/file/d/100zHo1b0Rui_rD_ikGr_K9I1Qije-4G3/view?usp=sharing) |

>  Place the weights file in the root folder before running the app.

---

## ⚙Setup Instructions

1. **Clone the Repo & Navigate**
2. **Install requirements:**
```bash
pip install -r requirements.txt
```

3. **Create `.env` with your OpenRouter API key:**
```
OPENROUTER_API_KEY=your_openrouter_key_here
```

4. **Run the App**
```bash
python app.py
```

5. Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## AI Advisor (via OpenRouter or Gemini)

- Backend uses OpenRouter API with models like `mistralai/mixtral-8x7b-instruct`
- Advises users based on prediction result
- Handles chat response errors + quota messages

Example context passed to AI:
```
The user's handwriting image was predicted as 'Static'.
```

---

## PDF Report Generator

- Built using `fpdf`
- Includes:
  - User name
  - Upload date & time
  - Prediction result + confidence
  - Uploaded image preview
  - AI Advisor’s response
  - Disclaimer

Output sample: `parkinson_report_YYYYMMDD_HHMMSS.pdf`

---

## 🛠️ Train the Model Yourself

```bash
python train_model.py
```

Your dataset must be organized like this:
```
dataset/
├── Static/
├── Dynamic/
```

> The script includes data augmentation, balanced sampling, classification report, and saves `model_weights.pth`.

---

## Requirements

- Python 3.7+
- PyTorch, torchvision
- Flask, fpdf
- scikit-learn
- dotenv

Install using:
```bash
pip install -r requirements.txt
```

---

## Disclaimer

> This tool is intended for educational and research purposes. It is not a certified medical device and should not be used for official diagnosis. Always consult a neurologist or healthcare provider for real medical guidance.

---

## Author

Developed by **Sakshi Mishra**  
B.Tech (Information Technology) — Parkinson Detection Project

---

