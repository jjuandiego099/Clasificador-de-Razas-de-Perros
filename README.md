# 🐾 Dog Breed Classifier

Aplicación web para clasificar razas de perros a partir de una imagen, construida con una CNN entrenada desde cero sobre el dataset Stanford Dogs.

---

## 📌 Descripción

Este proyecto entrena una red neuronal convolucional (CNN) para identificar **120 razas de perros** usando el [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). El modelo entrenado se despliega en una interfaz web construida con **Streamlit**.

---

## 🗂️ Estructura del proyecto

```
├── app.py               # Interfaz web con Streamlit
├── model.keras          # Modelo entrenado (no incluido en el repo)
├── parcial.ipynb        # Notebook de entrenamiento (Google Colab)
├── requirements.txt     # Dependencias
└── README.md
```

---

## 🧠 Arquitectura del modelo

El modelo es una CNN secuencial con **data augmentation** integrada:

**Data Augmentation:**
- Flip horizontal aleatorio
- Rotación aleatoria (±5%)
- Zoom aleatorio (±5%)
- Contraste aleatorio (±10%)
- Clip de valores al rango [0, 1]

**Capas del modelo:**

| Capa              | Detalles                        |
|-------------------|---------------------------------|
| Input             | (64, 64, 3)                     |
| Data Augmentation | RandomFlip, Rotation, Zoom, etc.|
| Conv2D            | 64 filtros, kernel 3×3, same    |
| LeakyReLU         | alpha = 0.1                     |
| MaxPooling2D      | 2×2                             |
| Conv2D            | 128 filtros, kernel 3×3, same   |
| LeakyReLU         | alpha = 0.1                     |
| MaxPooling2D      | 2×2                             |
| Flatten           | —                               |
| Dense             | 64 unidades, ReLU               |
| BatchNormalization| —                               |
| Dense             | 32 unidades, ReLU               |
| BatchNormalization| —                               |
| Dense (salida)    | 120 unidades, Softmax           |

**Compilación:**
- Optimizador: `Adam`
- Loss: `sparse_categorical_crossentropy`
- Métrica: `accuracy`

---

## 📦 Dataset

- **Nombre:** Stanford Dogs Dataset
- **Total de imágenes:** 20,580
- **Clases:** 120 razas de perros
- **Split:** 90% entrenamiento / 10% validación

**Preprocesamiento:**
1. Recorte con bounding box (anotaciones XML incluidas en el dataset)
2. Redimensionado a **64×64** píxeles
3. Normalización: división entre 255 → rango [0.0, 1.0]

---

## 🚀 Entrenamiento

El entrenamiento se realizó en **Google Colab** con las siguientes configuraciones:

- Epochs máximos: 50
- Batch size: 32
- **EarlyStopping:** paciencia de 5 épocas monitoreando `val_loss`
- **ModelCheckpoint:** guarda automáticamente el mejor modelo (`model.keras`)

---

## 🖥️ Aplicación web

La app permite subir una foto de un perro y obtener:
- La raza predicha con mayor confianza
- Barra de confianza visual
- Top 5 de razas más probables

### Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ El archivo `model.keras` debe estar en la misma carpeta que `app.py`.

---

## 📋 Requisitos

```
streamlit
numpy
Pillow
tensorflow
```

---

## ⚠️ Nota sobre compatibilidad del modelo

El modelo usa una capa `Lambda` con `tf.clip_by_value` dentro del bloque de data augmentation. Al cargar el modelo, es necesario inyectar `tf` en el scope de deserialización:

```python
import builtins
import tensorflow as tf
builtins.tf = tf

model = tf.keras.models.load_model("model.keras", custom_objects={"tf": tf}, compile=False)
```

---

## 👤 Autor

**Juan Diego Chaparro Garcia**  
Proyecto parcial — Redes Neuronales / Deep Learning
