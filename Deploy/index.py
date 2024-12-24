from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Define the CNN block for building the model
def cnn_block(in_ch, out_ch, pool=False):
    layers = [
        torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(out_ch),
        torch.nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(torch.nn.MaxPool2d(2))
    return torch.nn.Sequential(*layers)

# Define the model class
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

# Initialize the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and set it to evaluation mode
model = SCNN4(1, 4)  # Adjust the number of classes if necessary
model.load_state_dict(torch.load("scnn4.pth", map_location=device))
model = model.to(device)
model.eval()

# Set the image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Define the class labels
classes = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    
    # Transform the image and make a prediction
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, preds = torch.max(output, dim=1)
        prediction = classes[preds.item()]
    
    return jsonify({'class': prediction})

if __name__ == '__main__':
    app.run(debug=True)