# 🪐 Deep Hunter Exoplanet

🚀 Demo:
https://huggingface.co/spaces/Giotto/deep-hunter-exoplanet

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

Se crearon tres conjuntos de datos de dificultad incremental para evaluar el comportamiento del modelo y estudiar sus limitaciones.

La experimentación siguió un enfoque iterativo:

1. Validar si la representación visual contiene información suficiente.
2. Incrementar la dificultad con falsos positivos más similares.
3. Reducir la diferencia entre datos sintéticos y observaciones reales (*domain shift*).

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

## Experimento 3 — Adaptación Visual Estilo Kepler (*Kepler Style Dataset*)

Tras probar el modelo anterior con una curva real del telescopio Kepler, se detectó un problema de **domain shift**: las imágenes sintéticas utilizadas durante el entrenamiento eran visualmente diferentes a las observaciones astronómicas reales.

Por este motivo, se creó una tercera versión del dataset simulando características propias de las curvas Kepler:

**Características añadidas:**

* Mayor resolución temporal (más puntos por curva).
* Ruido instrumental más intenso.
* Variabilidad estelar de baja frecuencia.
* Artefactos verticales similares a discontinuidades instrumentales.
* Estilo visual más parecido a publicaciones astronómicas reales:
  * curva negra,
  * mayor densidad de puntos,
  * patrones menos suavizados.

El objetivo no era simplemente mejorar la métrica, sino comprobar si un dataset sintético más realista permitía mejorar la generalización.

**Resultados:**

| Métrica | Valor |
| --- | --- |
| Accuracy | 0.980 |
| Precision | 0.956 |
| Recall | 1.000 |
| F1-score | 0.978 |

El resultado más relevante es el **Recall = 1.0**.

Esto indica que, dentro del conjunto de validación, el modelo fue capaz de recuperar todos los candidatos planetarios.

En un sistema de búsqueda astronómica preliminar este comportamiento es especialmente interesante, ya que es preferible generar algunos falsos positivos adicionales antes que descartar posibles descubrimientos.

---

# 📈 Comparación Global de Experimentos

| Experimento | Objetivo | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Baseline | Validar pipeline CV | 1.000 | 1.000 | 1.000 | 1.000 |
| Hard Dataset | Evaluar robustez | 0.975 | 0.961 | 0.990 | 0.975 |
| Kepler Style | Reducir domain shift | 0.980 | 0.956 | 1.000 | 0.978 |

La evolución experimental muestra que el primer resultado perfecto no implicaba necesariamente un modelo más robusto. 

La introducción progresiva de ruido, falsos positivos difíciles y artefactos realistas permitió evaluar mejor las limitaciones del sistema.

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

Finalmente, se evaluó el sistema utilizando una imagen real de una curva de luz procedente de Kepler correspondiente al exoplaneta confirmado **Kepler-8b**.

El objetivo era comprobar si los patrones aprendidos con datos sintéticos podían transferirse parcialmente a datos astronómicos reales.

## Modelo entrenado con Hard Dataset

Resultado inicial:

|Clase|Probabilidad|
|-|-|
|Planet|57%|
|False Positive|43%|

Aunque el modelo clasificó correctamente la curva como candidata planetaria, la baja confianza reveló un problema de **domain shift**.

## Modelo entrenado con Kepler Style Dataset

Tras adaptar la generación sintética para parecerse más al dominio real:

|Clase|Probabilidad|
|-|-|
|Planet|100%|
|False Positive|0%|

La mejora de confianza sugiere que parte de la incertidumbre inicial estaba causada por diferencias visuales entre los datos de entrenamiento y las curvas reales.

Este resultado no implica la confirmación automática de exoplanetas, pero demuestra la importancia de diseñar datasets representativos del dominio objetivo.

---

## Domain Shift detectado

Diferencias principales:

**Dataset inicial**

* Curvas limpias.
* Fondo homogéneo.
* Ruido controlado.
* Representación simplificada.

**Datos reales Kepler**

* Mayor ruido instrumental.
* Variabilidad estelar.
* Artefactos visuales.
* Mayor complejidad.

La solución propuesta fue incorporar estas características al proceso de generación sintética.

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
│   └── app.py                      # Aplicación web interactiva (Gradio)
├── data/
│   ├── images/                     # Dataset Experimento 1 (Baseline)
│   ├── images_hard/                # Dataset Experimento 2 (Hard)
│   ├── images_kepler_style/         # Dataset Experimento 3 (Estilo Kepler Real)
│   └── test_real_kepler/           # Muestras reales de Kepler para validación
├── models/
│   ├── resnet18_exoplanetas.pt     # Peso del modelo Baseline
│   ├── resnet18_exoplanetas_images_hard.pt
│   └── resnet18_exoplanetas_images_kepler_style.pt
├── notebooks/                      # Jupyter Notebooks de desarrollo y análisis
│   ├── 01_dataset.ipynb
│   ├── 02_training.ipynb
│   └── 03_results.ipynb
├── reports/figures/                # Matrices de confusión de los 3 experimentos
│   ├── confusion_matrix.png
│   ├── confusion_matrix_images_hard.png
│   └── confusion_matrix_images_kepler_style.png
├── src/                            # Código fuente del proyecto
│   ├── data/                       # Scripts de generación de datos sintéticos
│   │   ├── generate_images.py
│   │   ├── generate_images_hard.py
│   │   └── generate_images_kepler_style.py
│   └── models/                     # Scripts de entrenamiento e inferencia
│       ├── predict.py
│       ├── train.py
│       ├── train_hard.py
│       └── train_kepler_style.py
├── pyproject.toml                  # Configuración del entorno y dependencias
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

# ⚠️ Limitaciones

Aunque el modelo consigue buenos resultados, existen varias limitaciones:

- El entrenamiento se realiza sobre datos sintéticos.
- La predicción depende de la representación visual utilizada.
- El modelo aprende patrones morfológicos, pero no realiza una validación astrofísica completa.
- Confirmar un exoplaneta requiere análisis adicionales como periodicidad orbital, profundidad del tránsito y validación estadística.

Este sistema debe entenderse como una herramienta preliminar de filtrado.