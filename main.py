from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="EyeCare AI Backend by Sonam")

# CORS middleware zaroori hai taaki HTML frontend Python se baat kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Production me "*" ki jagah apni website ka URL dalte hain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. MODEL SETUP ---
device = torch.device('cpu') # Server par hum CPU use karenge
print("Loading EfficientNet-B3 Model...")

# Model define karna
model = models.efficientnet_b3()
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 8) # 8 diseases

# Saved weights load karna (Error handling ke sath)
try:
    model.load_state_dict(torch.load('odir_efficientnet_b3.pth', map_location=device))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Model file 'odir_efficientnet_b3.pth' not found. Ensure it is in the same folder.")

model.eval()

# Disease list
disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

# --- 2. IMAGE PREPROCESSING ---
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. API ENDPOINT ---
@app.post("/predict")
async def predict_eye(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        img_tensor = val_transforms(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output).squeeze().tolist()
        
        # Format response
        results = {}
        for i, disease in enumerate(disease_names):
            results[disease] = {
                "percentage": round(probs[i] * 100, 2),
                "detected": probs[i] > 0.5 # True agar 50% se zyada hai
            }
            
        return results
        
    except Exception as e:
        return {"error": str(e)}

# Render par run karne ke liye ek root endpoint
@app.get("/")
def read_root():
    return {"message": "Sonam's EyeCare AI API is running successfully!"}