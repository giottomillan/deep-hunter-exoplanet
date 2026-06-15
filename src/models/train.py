from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

#---------------
#CONFIGURACIONES DE PARÁMETROS
#---------------
# configuraciones
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "images"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

#hiperparámtros de la red
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

#definimos el dispositivo
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

print(f"Usando dispositivo: {DEVICE}")

#--------------
#TRANSOFRMACIONES
#--------------
# transformaciones y data augmentation
transform = transforms.Compose(
    [
        #redimensionamos porque ResNet18 porque fue diseñado para procesar imágenes de 224x224 píxeles
        transforms.Resize((224, 224)),
        #aplicamos rotación para que la red no se vuelva perezosa
        transforms.RandomRotation(degrees=5),
        #convertimos los píxles de las imágenes en matrices para los tensores de pytorch
        transforms.ToTensor(),
        #ahora normalizamos
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

#cargamos y dividimos los datos
#indicamos la carpeta donde están las clases para que pytorch las identifique
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print(dataset.classes)


#hacemos el split en entranimiento y validación 
train_size = int(0.8 * len(dataset))

val_size = (len(dataset) - train_size)

train_ds, val_ds = random_split(dataset, [train_size,  val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


#--------------
#MODELO RESNET
#--------------

#definimos el modelo de Resnet que ya está preentrenada
model = models.resnet18(weights="DEFAULT")

#cortamos la última capa para que haga clasificación binaria y tenga dos salidas
model.fc = nn.Linear(model.fc.in_features, 2)

#inicializamos el modelo
model = model.to(DEVICE)

#definimos la función de pérdida para medir el error como criterio de optimización
criterion = nn.CrossEntropyLoss()

#definimos el optimizador ADAM
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#-----------------
#ENTRENAMIENTO
#-----------------

for epoch in range(EPOCHS):
    
    model.train()
    
    total_loss = 0
    
    for images, labels in train_loader:
        
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        #reseteamos los gradientes porque pytorch los acumula por defecto
        optimizer.zero_grad()
        
        #pasamos las imágenes por la red
        outputs = model(images)
        
        #calculamos el loss
        loss = criterion(outputs, labels)
        
        #hacemos el backpropagation
        loss.backward()
        
        #ajustamos los pesos con el optimizador
        optimizer.step()
        
        total_loss += loss.item()
        
    print(
        f"Epoca {epoch+1}/{EPOCHS}"
        f"Pérdida = {total_loss:.4f}"
    )
    
#-------------
#EVALUACIÓN
#-------------
#hacemos la evaluación del modelo
model.eval()

predictions = []
targets = []

with torch.no_grad():
    
    for images, labels in val_loader:
        
        images = images.to(DEVICE)
        
        outputs = model(images)
        
        predicted = outputs.argmax(dim=1)
        
        #movemos el tensor de predicciones a la cpu para que scikit learn pueda calcular las métricas
        #y desempaquetamos con extend para tener una lista plana
        predictions.extend(predicted.cpu().numpy())
        
        targets.extend(labels.numpy())
        
        
        
print("RESULTS:")
print("Precisión/Accuracy:", accuracy_score(targets, predictions))
print("Precisión:", precision_score(targets, predictions))
print("Recall:", recall_score(targets, predictions))
print("F1:", f1_score(targets, predictions))

#hacemos una confusion matrix
#la calculamos
cm = confusion_matrix(targets, predictions)
#definimos la figura
plt.figure(figsize=(5, 4))
#la pintamos
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=dataset.classes,
    yticklabels=dataset.classes
)
plt.xlabel("Predicciones")
plt.ylabel("Real")
plt.tight_layout()
#salvamos la figura
plt.savefig(REPORT_DIR/"confusion_matrix.png")
print("Figura de las predicciones salvada")

#salvamos el modelo
torch.save(model.state_dict(), MODEL_DIR/"resnet18_exoplanetas.pt")
print("Modelo salvado!")
    
    

