from pathlib import Path
import gradio as gr
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

#CONGIGURACIONES DE PARÁMETROS DE RUTA
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_exoplanetas_images_kepler_style.pt"

CLASS_NAMES = ["false_positive", "planet"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FUNCIONES
#definimos las funciones de carga del modelo para que puedan ser independienes de la carpeta scr
#para cargar el modelo
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES),)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,))
    model.to(DEVICE)
    model.eval()
    return model
#cargamos el modelo para la app
model = load_model()

#aplicamos las trasformaciones pertinentes
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ]
)

#función para predecir
def predict(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        return {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    
description = """
Upload a light curve image and the model will estimate whether it looks like an exoplanet transit candidate or a false positive.
Sube una imagen de curva de luz y el modelo estimará si se trata de un posible tránsito de exoplaneta o de un falso positivo.


This is a computer vision proof of concept. It does not scientifically confirm exoplanets.
Esta es una prueba de concepto de visión artificial. No confirma científicamente la existencia de exoplanetas.
"""

#definimos el objeto de gradio
demo=gr.Interface(
    theme="soft",
    #para que cuando el usuario suba una foto y le de a neviar tome ese dato y lo pase como armgumento para el predict
    fn = predict,
    
    #configuramos una caja arrastrable para que el usuario pueda subir una imagen y lo trasforme a PIL
    inputs=gr.Image(type="pil", label="Light curve image / Imagen de curva de luz",),
    
    #para que se puedan pintar una barras con el objeto dict que devuelve la función
    outputs=gr.Label(num_top_classes=2, label="Prediction / Predicción",),
    
    #definimos titulos, descripción y colocamos ejemplos
    title="Deep Hunter Exoplanet",
    description=description,
    examples=[
        str(PROJECT_ROOT / "data" / "images_hard" / "planet" / "planeta_hard_0000.png"),
        str(PROJECT_ROOT / "data" / "images_hard" / "false_positive" / "falso_positivo_hard_0000.png"),
    ],    
)

if __name__ == "__main__":
    demo.launch()