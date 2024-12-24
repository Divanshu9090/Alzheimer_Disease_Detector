import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# Load the trained model
class SCNN4(torch.nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.cnn1 = cnn_block(in_ch, 8, pool=False)
        self.cnn2 = cnn_block(8, 16, pool=True)
        self.cnn3 = cnn_block(16, 32, pool=True)
        self.cnn4 = cnn_block(32, 64, pool=True)
        self.last = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64*31*62, n_classes)
        )

    def forward(self, xb):
        out = self.cnn1(xb)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        out = self.last(out)
        return out

def cnn_block(in_ch, out_ch, pool=False):
    layers = [
        torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(out_ch),
        torch.nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(torch.nn.MaxPool2d(2))
    return torch.nn.Sequential(*layers)

# Define the function for prediction
def predict_label(img, model, classes):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return classes[preds[0].item()]

# Load the model and set it to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SCNN4(1, 4)  # Adjust the number of classes if different
model.load_state_dict(torch.load("scnn4.pth", map_location=device))
model = model.to(device)
model.eval()

# Set up the image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Set the list of classes
classes = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']  # Replace with your actual class names

# Streamlit interface
st.title("Image Classification Web App")
st.write("Upload an image and let the model predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform the image and predict
    img_tensor = transform(image).to(device)
    label = predict_label(img_tensor, model, classes)
    
    st.write(f"**Prediction:** {label}")
