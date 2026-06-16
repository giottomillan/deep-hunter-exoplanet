# 🪐 Deep Hunter Exoplanet

Deep Hunter Exoplanet es un proyecto de Visión por Computador enfocado en la detección de patrones similares a tránsitos de exoplanetas a partir de curvas de luz astronómicas.

El proyecto explora un enfoque alternativo:

> ¿Puede una red neuronal convolucional identificar patrones de tránsito planetario si las series temporales astronómicas se transforman en imágenes?

En lugar de trabajar directamente con series temporales numéricas, esta primera versión convierte las curvas de luz en representaciones visuales y aplica aprendizaje por transferencia (*transfer learning*) utilizando técnicas modernas de clasificación de imágenes.

---

# 🌌 Motivación

La mayoría de los exoplanetas conocidos han sido descubiertos mediante el **método de tránsito**.

Cuando un planeta pasa por delante de su estrella anfitriona, el brillo observado disminuye ligeramente:

```text
Brillo Normal
─────────────
             \__/
               ↑
            Tránsito

```

Detectar estos eventos es complejo debido a que:

* La profundidad del tránsito puede ser extremadamente pequeña.
* La variabilidad estelar introduce ruido en la señal.
* Los artefactos instrumentales distorsionan las mediciones.
* Otros eventos astronómicos pueden imitar el comportamiento de los planetas.

El objetivo de este proyecto no es confirmar científicamente exoplanetas, sino explorar si el Deep Learning puede actuar como un sistema de cribado inicial.

---

# 🎯 Objetivo del Proyecto

El objetivo de esta primera versión es construir un pipeline completo de Visión por Computador:

```text
 Curva de Luz
       ↓
Generación de Imagen
       ↓
CNN de Transfer Learning
       ↓
Candidato a Planeta  |  Falso Positivo

```

El sistema recibe la representación en imagen de una curva de luz y predice si el patrón se asemeja a:

* 🪐 Tránsito Planetario (*Planet*)
* ❌ Falso Positivo (*False Positive*)

---

# 🛰️ Generación del Dataset

Se creó un conjunto de datos sintético para simular observaciones astronómicas reales. Se generaron dos clases:

## 🪐 Planeta (*Planet*)

Los candidatos a planeta contienen:

* Una única disminución de brillo similar a un tránsito.
* Posición del tránsito aleatoria.
* Profundidad del tránsito variable.
* Duración del tránsito variable.
* Ruido observacional gaussiano.

Comportamiento de ejemplo:

```text
──────────
          \__/

```

## ❌ Falso Positivo (*False Positive*)

Los falsos positivos incluyen:

* Ruido aleatorio puro.
* Variabilidad estelar (sinuosidades).
* Comportamiento de binarias eclipsantes.
* Artefactos artificiales.

Ejemplo:

```text
──\_/──────\_/──

```

Esta configuración evita que el modelo aprenda una regla trivial como *"caída de brillo = planeta"*, obligándolo a entender la morfología de la curva.

---

# 🧪 Experimentos

Se crearon dos conjuntos de datos para evaluar el modelo de forma progresiva.

## Experimento 1 — Dataset Base (Baseline)

**Objetivo:** Validar si el pipeline funciona bajo condiciones controladas.

**Dataset:**

* 500 imágenes de planetas
* 500 imágenes de falsos positivos

**Resultado:**

| Métrica | Valor |
| --- | --- |
| Accuracy | 1.000 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1-score | 1.000 |

Aunque el experimento fue exitoso, demostró que el conjunto de datos inicial era demasiado fácil de separar.

---

## Experimento 2 — Dataset Difícil (Hard Dataset)

Para evaluar la robustez del modelo, se diseñó un dataset con mayor complejidad.

**Complejidad adicional añadida:**

* Tránsitos mucho más débiles (menor profundidad).
* Mayor nivel de ruido gaussiano.
* Falsos positivos visualmente muy similares a los planetas.
* Caídas de brillo asimétricas.

**Resultados:**

| Métrica | Valor |
| --- | --- |
| Accuracy | 0.975 |
| Precision | 0.961 |
| Recall | 0.990 |
| F1-score | 0.975 |

El descenso en el rendimiento sugiere un escenario de evaluación mucho más realista.

El modelo prioriza el **recall (sensibilidad)**, lo que significa que se pierden muy pocos candidatos a planeta. Para los sistemas de cribado astronómico, este comportamiento es el ideal:

* **Falso positivo:** puede ser revisado y descartado por expertos más tarde.
* **Falso negativo:** se traduce en un planeta potencial que se pierde para siempre.

---

# 🧠 Modelo

El clasificador hace uso de aprendizaje por transferencia (*transfer learning*).

**Arquitectura:**

* ResNet18 preentrenada en ImageNet.
* Capa de clasificación final (*head*) modificada.
* Salida binaria.

**Clases:**

* `false_positive`
* `planet`

**Configuración del entrenamiento:**

| Parámetro | Valor |
| --- | --- |
| Tamaño de imagen | 224x224 |
| Optimitzador | Adam |
| Learning rate | 1e-4 |
| Epochs | 5 |
| Batch size | 32 |

Se seleccionó *transfer learning* debido a que el tamaño del dataset es limitado, el entrenamiento es drásticamente más rápido y permite reutilizar los filtros convolucionales ya optimizados para extraer bordes y texturas.

---

# 📊 Evaluación

El modelo se evalúa utilizando:

* Accuracy (Exactitud)
* Precision (Precisión)
* Recall (Sensibilidad)
* F1-score
* Matriz de confusión

Se presta especial atención al *recall* debido al alto coste crítico que supone omitir un posible candidato a planeta.

---

# 🌍 Experimento con Datos Reales

Se sometió al modelo a una prueba utilizando una imagen de una curva de luz real extraída del telescopio espacial Kepler (correspondiente al exoplaneta confirmado **Kepler-8b**).

**Predicción obtenida:**

* **Planet (Planeta):** 57%
* **False Positive (Falso Positivo):** 43%

El modelo identificó correctamente al candidato como un patrón planetario, aunque el nivel de confianza disminuyó. Esto reveló una limitación técnica clave:

## Domain Shift (Desplazamiento de Dominio)

* **Imágenes de entrenamiento:** Sintéticas, fondo limpio, ruido controlado, formato estandarizado y un solo tránsito centrado.
* **Imágenes astronómicas reales:** Mayor nivel de ruido base, estilos de visualización diferentes, artefactos del instrumento y líneas de cuadrículas/ejes.

Las futuras versiones reducirán esta brecha generando datos sintéticos más realistas e incorporando observaciones reales de las misiones Kepler y TESS dentro del set de entrenamiento.

---

# 🚀 Despliegue

Se desarrolló una aplicación web interactiva utilizando **Gradio**. El usuario puede:

1. Subir la imagen de una curva de luz.
2. Ejecutar la inferencia del modelo en tiempo real.
3. Recibir la clase predicha junto con su probabilidad de confianza mediante barras gráficas.

**Ejemplo de flujo:**

* **Entrada:** Imagen recortada de la curva de luz.
* **Salida:** `planet` → 0.97 | `false_positive` → 0.03

---

# 🏗️ Estructura del Proyecto

```text
deep-hunter-exoplanet/
├── app/
│   └── app.py
│
├── data/
│   ├── images/
│   ├── images_hard/
│   └── raw/
│
├── models/
│   └── resnet18_exoplanetas.pt
│
├── reports/
│   └── figures/
│       └── confusion_matrix.png
│
├── src/
│   ├── data/
│   │   ├── generate_images.py
│   │   └── generate_images_hard.py
│   │
│   └── models/
│       ├── train.py
│       └── predict.py
│
└── README.md

```

---

# 🔭 Trabajo Futuro

Este enfoque basado en Visión por Computador representa la primera etapa de una investigación más amplia.

## Datasets más realistas

* Integración de curvas de luz reales de la misión Kepler.
* Incorporación de observaciones del satélite TESS.
* Modelado avanzado del ruido instrumental del telescopio.

## Deep Learning Nativo para Series Temporales

Evolucionar desde la clasificación de imágenes hacia modelos que procesen la señal numérica original de forma nativa:

* Redes Convolucionales 1D (1D CNN).
* Transformers aplicados a series temporales.
* Modelos temporales recurrentes.

## Arquitectura Inspirada en VARnet

Diseño de una arquitectura futura inspirada en el modelo VARnet:

* Rama convolucional temporal (1D CNN).
* Extracción de características en el dominio de la frecuencia mediante Fourier.
* Descomposición basada en Wavelets.
* Fusión de características extraídas para la clasificación final.

---

# Referencias

* Misión Kepler de la NASA
* Misión TESS de la NASA
* Archivo de Exoplanetas de la NASA (NASA Exoplanet Archive)
* He et al. (2015) — Deep Residual Learning for Image Recognition
* Paz, M. (2024). VARnet: A Submillisecond Fourier and Wavelet-based Model to Extract Variable Candidates from the NEOWISE Single-exposure Database.

```