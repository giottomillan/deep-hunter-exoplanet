from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

#CONFIGURACIÓN PARÁMETROS
#para que pueda llegar en cualquier sitio que se corra este código a las carpetas pertinentes
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#indicamos la ruta del modelo entrenado, en este aso el que ha sido entrenado con las imágenes complejas
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_exoplanetas_images_hard.pt"

#especificamos los nombres de las clases
CLASS_NAME = [
    "false_positive",
    "planet",
]

#definimos el dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#FUNCIONES

#para configurar y cargar el modelo
def load_model():
    #none porque usaremos el nuestro
    model = models.resnet18(weights=None)
    
    #modificamos la última capa para que sea binaria la clasificación como nuestro modelo
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAME),)
    
    #para que se puedan adapatar los pesos en caso de que no se tenga GPU
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE,))
    
    model.to(DEVICE)
    #apagamos capas de entrenamiento porque ya no está aprendiendo
    model.eval()
    
    return model

#para preparar la imagen de entrada en el formato que espera el modelo con las traformaciones que se aplicaron en el entrenamiento
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
            #usamos los valores usados como constantes para modelos como resnet
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ]
    )

#para la predicción de las clases de las imágenes
def predict_image(image_path: str):
    #modelo y trasformaciones
    model = load_model()
    transform = get_transform()
    
    #forzamos el modo RGB que es lo que resnet18 exige
    image = Image.open(image_path).convert("RGB")
    
    #usamos unsqueeze para que la lista tenga la dimensión del lote
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    #desactivamos el cálculo de gradientes y backpropragation
    with torch.no_grad():
        outputs = model(tensor)
        #función softmax para que sean probabilidades que sumen 100%
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        #construímos el diccionario para quedarnos con el máximo
        results = {
            CLASS_NAME[i]: float(probabilities[i]) for i in range(len(CLASS_NAME))
        }
        #nos quedamos con la que tenga el mayor valor
        predicted_class = max(results, key=results.get)
        
        return predicted_class, results
    
#bloque de ejecución
if __name__ == "__main__":
    test_image = ( PROJECT_ROOT / "data" / "images_hard" / "planet" / "planeta_hard_0000.png" )
    #hacemos la predicción con la imagen de prueba pasando la ruta en str
    predicted_class, results = predict_image(str(test_image))
    
    print(f"Imagen: {test_image}")
    print(f"Predicción: {predicted_class}")
    print(f"Probabilidad:")
    for class_name, probability in results.items():
        print(f" {class_name}. {probability:.4f}")