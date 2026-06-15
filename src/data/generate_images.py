from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#establecemos las constantes del proyecto para la ruta de salida de las imágenes generadas y los parámetros de generación de datos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "images"

N_SAMPLES_PER_CLASS = 500
N_POINTS = 300
SEED = 42

#creamos las funciones para generar los datos de las clases y guardarlas en imágenes
def create_planet_curve(n_points: int = N_POINTS) -> np.ndarray:
    """genera una curva sintética de luz con tránsido y caida de luz, simulando un planeta que pasa frente a una estrella"""
    x = np.linspace(0, 1, n_points)
    
    flux = np.ones(n_points)
    
    #parámetros random de tránsito
    center = np.random.uniform(0.35, 0.65)
    width = np.random.uniform(0.035, 0.09)
    depth = np.random.uniform(0.014, 0.08)
    
    transit = depth * np.exp(-0.5 * ((x - center) / width) ** 2)
    flux -= transit
    
    #añadimos ruidos esterles
    noise = np.random.normal(0, 0.006, n_points)
    trend = np.random.normal(0, 0.003) * (x - 0.5)
    
    return flux + noise + trend

def create_false_positive_curve(n_points: int = N_POINTS) -> np.ndarray:
    """genera una curva sintética de luz con variabilidad estelar, simulando una estrella variable sin tránsito"""
    x = np.linspace(0, 1, n_points)
    
    curve_type = np.random.choice(["noise", "variable_star", "eclipsing_binary"])
    
    if curve_type == "noise":
        flux = np.ones(n_points)
        flux += np.random.normal(0, 0.02, n_points)
        
    elif curve_type == "variable_star":
        amplitude = np.random.uniform(0.01, 0.06)
        frequency = np.random.uniform(1.5, 5.0)
        phase = np.random.uniform(0, 2 * np.pi)
        
        flux = 1 + amplitude * np.sin(2*np.pi * frequency * x + phase)
        flux += np.random.normal(0, 0.008, n_points)
        
    else:
        flux = np.ones(n_points)
        
        center_1 = np.random.uniform(0.25, 0.45)
        center_2 = np.random.uniform(0.60, 0.80)
        
        width_1 = np.random.uniform(0.025, 0.06)
        width_2 = np.random.uniform(0.025, 0.06)
        
        depth_1 = np.random.uniform(0.05, 0.18)
        depth_2 = np.random.uniform(0.03, 0.12)
        
        eclipse_1 = depth_1 * np.exp(-0.5 * ((x - center_1) / width_1) ** 2)
        eclipse_2 = depth_2 * np.exp(-0.5 * ((x - center_2) / width_2) ** 2)
        
        flux -= eclipse_1 + eclipse_2
        flux += np.random.normal(0, 0.008, n_points)
        
    return flux

def save_curve_as_image(flux: np.ndarray, output_path: Path) -> None:
    """salvar la curva como una imagen limpia para clasificacion computer vision"""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    
    ax.plot(flux, linewidth=1.8)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    
    # fijamos el límite del eje y para que se pueda ver mejor una caída de luz profunda
    ax.set_ylim(0.75, 1.10)
    
    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches = "tight", pad_inches=0)
    plt.close(fig)
    
def generate_dataset() -> None:
    "función global para generar el dataset"
    np.random.seed(SEED)
    
    planet_dir = OUTPUT_DIR / "planet"
    false_positive_dir = OUTPUT_DIR / "false_positive"
    
    planet_dir.mkdir(parents=True, exist_ok=True)
    false_positive_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(N_SAMPLES_PER_CLASS), desc="Generando imágenes de planetas"):
        flux = create_planet_curve()
        save_curve_as_image(flux, planet_dir / f"planeta_{i:04d}.png")
        
    for i in tqdm(range(N_SAMPLES_PER_CLASS), desc="Generando imágenes de falsos positivos"):
        flux = create_false_positive_curve()
        save_curve_as_image(flux, false_positive_dir / f"falso_positivo_{i:04d}.png")
        
    print("Dataset generado correctamente")
    print(f"Imágenes de planetas: {len(list(planet_dir.glob("*.png")))}")
    print(f"Imágenes de falsos positivos: {len(list(false_positive_dir.glob("*.png")))}")
    print(f"Directorio de salida: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()