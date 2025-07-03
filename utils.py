import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

Device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    classes = ['aneurysm', 'cancer', 'tumor']
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5, inplace=True),
        nn.Linear(1024, 16, bias=False),
        nn.BatchNorm1d(16),
        nn.ReLU(inplace=True),
        nn.Linear(16, 3)
    )
    model.load_state_dict(torch.load("model/Model.pth", map_location=Device))
    model.to(Device)
    model.eval()
    return model, classes

def load_tumor_model():
    tumor_classes = ['glioblastoma', 'meningioma', 'pituitary']
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5, inplace=True),
        nn.Linear(1024, 16, bias=False),
        nn.BatchNorm1d(16),
        nn.ReLU(inplace=True),
        nn.Linear(16, len(tumor_classes))
    )
    model.load_state_dict(torch.load("model/TumerModel.pth", map_location=Device))
    model.to(Device)
    model.eval()
    return model, tumor_classes

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_image(image_path, model, classes):
    image_tensor = transform_image(open(image_path, 'rb')).to(Device)
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return classes[pred], probs[0][pred].item()

def predict_tumor_type(image_path, tumor_model, tumor_classes):
    image_tensor = transform_image(open(image_path, 'rb')).to(Device)
    outputs = tumor_model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return tumor_classes[pred], probs[0][pred].item()
