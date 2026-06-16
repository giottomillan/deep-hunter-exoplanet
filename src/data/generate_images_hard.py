from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#CONFIGURACION

#parámetros de ruta
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "images_hard"

#parámetros del código
N_SAMPLE_PER_CLASS = 500
N_POINTS = 300
SEED = 123

#FUNCIONES

#para crear una curva por tránsito de planeta más difícil
def create_planet_curve(n_points: int = N_POINTS) -> np.ndarray:
    x = np.linspace(0, 1, n_points)
    
    flux = np.ones(n_points)
    
    #usamos una distribución uniforme entre los rangos para simular paso por el centro, tamaño y velocidad
    center = np.random.uniform(0.30, 0.70)
    width = np.random.uniform(0.020, 0.075)
    depth = np.random.uniform(0.008, 0.045)
    
    #cuando hay tránsito restamos
    transit = depth * np.exp(-0.5 * ((x - center) / width) ** 2)
    flux -= transit
    
    #para tendencia y ruido usamos una distribución de campana de gauss
    trend = np.random.normal(0, 0.010) * (x - 0.5)
    noise = np.random.normal(0, 0.010, n_points)
    
    #cuando hay ruido y tendencia sumamos
    flux += trend + noise
    return flux 

#función para crear falsos positivos y engañar a la red
def create_false_positive_curve(n_points: int = N_POINTS) -> np.ndarray:
    x = np.linspace(0, 1, n_points)
    
    #definimos unos rango y tipos de curva
    curve_type = np.random.choice(
        [
            "shallow_single_dip",
            "asymmetric_dip",
            "variable_star",
            "noise_with_artifact"
        ],
        p=[0.40, 0.25, 0.20, 0.15]
    )
    
    flux = np.ones(n_points)
    
    if curve_type == "shallow_single_dip":
        #media para el cálculo de la campana de gauss
        center = np.random.uniform(0.30, 0.70)
        #desviación estándar
        width = np.random.uniform(0.025, 0.090)
        #altura
        depth = np.random.uniform(0.008, 0.050)
        
        #formula de la campana usando exp para darle la forma curva con la exponencial 
        dip = depth * np.exp(-0.5 * ((x - center) / width) ** 2)
        #para que sea inversa
        flux -= dip
    
    
    elif curve_type == "asymmetric_dip":
        #para crear curvas asimétricas y maliciosas definimos parámetros
        center = np.random.uniform(0.30, 0.70)
        #curva más estrecha y rápida
        width_left = np.random.uniform(0.020, 0.070)
        #curva más ancha
        width_right = np.random.uniform(0.040, 0.120)
        depth = np.random.uniform(0.010, 0.060)
        
        #inicializamos el lienzo
        dip = np.zeros_like(x)
        
        #máscara booleana
        left = x < center
        right = x >= center
        
        #dibujamos las curvas
        dip[left] = depth * np.exp(-0.5 * ((x[left] - center) / width_left) ** 2)
        dip[right] = depth * np.exp(-0.5 * ((x[right] - center) / width_right) ** 2)
        
        flux -= dip
        
    elif curve_type == "variable_star":
        amplitude = np.random.uniform(0.008, 0.040)
        frequency = np.random.uniform(1.0, 3.5)
        phase = np.random.uniform(0, 2 * np.pi)
        
        #dibujamos una onda senoidal para replicar la variación que produce una estrella variable
        flux += amplitude * np.sin(2 * np.pi * frequency * x + phase)
    #para ruidos con artefactos
    else:
        artifact_position = np.random.randint(40, n_points - 40)
        artifact_width = np.random.randint(5, 25)
        artifact_depth = np.random.uniform(0.015, 0.060)
        
        #para simular el error de algún artefacto generamos una caída repentina
        flux[artifact_position: artifact_position +artifact_width] -= artifact_depth
        
    trend = np.random.normal(0, 0.012) * (x - 0.5)
    noise = np.random.normal(0, 0.012, n_points)
    
    flux += trend + noise
    
    return flux
        
#función para guardar la imagen
def save_curve_as_image(flux: np.ndarray, output_path: Path) -> None:
    #para que tenga una resolución de 300x300 pixeles
    fig, ax = plt.subplots(figsize = (3, 3), dpi=100)
    
    ax.plot(flux, linewidth=1.8)
    
    #para dejarlo sin números y que no confunda a la red
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    #fijamos el límite para que no haya autozoom
    ax.set_ylim(0.82, 1.08)
    
    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
#función global para generar el dataset y las imágenes
def generate_dataset() -> None:
    np.random.seed(SEED)
    
    planet_dir = OUTPUT_DIR / "planet"
    false_positive_dir = OUTPUT_DIR / "false_positive"
    
    planet_dir.mkdir(parents=True, exist_ok=True)
    false_positive_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(N_SAMPLE_PER_CLASS), desc="Generando planetas hard"):
        flux = create_planet_curve()
        save_curve_as_image(flux, planet_dir / f"planeta_hard_{i:04d}.png")
        
    for i in tqdm(range(N_SAMPLE_PER_CLASS), desc="Generando flasos positivos hard"):
        flux = create_false_positive_curve()
        save_curve_as_image(flux, false_positive_dir / f"falso_positivo_hard_{i:04d}.png")
        
    print("Dataset hard generado correctamente")
    print(f"Total de planetas: {len(list(planet_dir.glob('*.png')))}")
    print(f"Total de falsos positivos: {len(list(false_positive_dir.glob('*.png')))}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    
if __name__ == "__main__":
    generate_dataset()
    
        
        
        
        