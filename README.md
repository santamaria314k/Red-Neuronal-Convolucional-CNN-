# 🌼 Clasificador de flores con una CNN (TensorFlow/Keras + Streamlit)

Proyecto de Deep Learning: una **Red Neuronal Convolucional** entrenada desde cero para clasificar fotografías de flores en **5 clases** (margarita, diente de león, rosa, girasol y tulipán), con una **aplicación Streamlit** que carga el modelo entrenado, recibe una imagen y muestra la **clase predicha y su nivel de confianza**.

Prueba de ejecusionde la app 
<img width="1385" height="730" alt="rosa ev" src="https://github.com/user-attachments/assets/efd6e1b1-87dd-44a6-ad72-454e7e937a1d" />
<img width="1365" height="728" alt="tulipan ev" src="https://github.com/user-attachments/assets/0e4227af-4a9f-46fa-a3a8-e116b77d7d73" />
<img width="1351" height="726" alt="girasol ev" src="https://github.com/user-attachments/assets/f8ecfee0-f9c8-4000-a814-1279d7517bfe" />
<img width="1368" height="721" alt="avion ev" src="https://github.com/user-attachments/assets/cae49f6b-ce7e-432a-bcf0-4a0fa73d5334" />




## 1. Dataset
**flower_photos** (publicado por el equipo de TensorFlow): fotos de flores organizadas en 5 carpetas.
Se descarga automáticamente desde `https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz`.
No se incluye en el repositorio (`data/` está en `.gitignore`).

| Clase | Español |
|---|---|
| daisy | Margarita |
| dandelion | Diente de león |
| roses | Rosa |
| sunflowers | Girasol |
| tulips | Tulipán |

## 2. Estructura del repositorio
```
.
├── notebooks/
│   └── 01_cnn_flores_experimentacion.ipynb   # Experimentación completa (Google Colab)
├── src/
│   ├── config.py      # Hiperparámetros y rutas
│   ├── data.py        # Descarga, partición train/val/test y pipeline tf.data
│   ├── model.py       # Arquitectura de la CNN + data augmentation
│   ├── train.py       # Entrenamiento y guardado del modelo
│   ├── evaluate.py    # Evaluación en test (métricas y figuras)
│   ├── predict.py     # Predicción de una imagen (CLI y reutilizado por Streamlit)
│   └── plots.py       # Gráficas
├── app/
│   └── streamlit_app.py                      # Aplicación web
├── models/
│   ├── flower_cnn.keras                      # Modelo entrenado
│   └── class_names.json                      # Orden de las clases
├── reports/
│   ├── metrics.json  history.json            # Métricas e historial de entrenamiento
│   └── figures/                              # Gráficas generadas
├── requirements.txt
└── README.md
```

## 3. Metodología
1. **Exploración:** balance de clases, ejemplos, tamaños de imagen, integridad de archivos y estadísticas de píxeles.
2. **Preprocesamiento:** decodificación RGB, redimensionado a 180×180 y normalización `/255` (capa `Rescaling` dentro del modelo, así la app no puede olvidarla).
3. **División estratificada** con semilla fija: 70 % entrenamiento / 15 % validación / 15 % prueba.
4. **Data Augmentation** (solo en entrenamiento): volteo horizontal, rotación, zoom y contraste.
5. **Arquitectura:** 4 bloques `[Conv→BN→ReLU]×2→MaxPool` (32-64-128-256 filtros) → GlobalAveragePooling → Dropout → Dense(128) → Dropout → Softmax(5).
6. **Entrenamiento:** Adam, `sparse_categorical_crossentropy`, EarlyStopping (restaura los mejores pesos) y ReduceLROnPlateau.
7. **Evaluación:** accuracy, precisión/recall/F1 por clase, F1 macro y matriz de confusión sobre el conjunto de prueba.
8. **Predicciones:** 12 imágenes de prueba con clase real, clase predicha y probabilidad.

## 4. Resultados
Resultados reales sobre el **conjunto de prueba** (551 imágenes que el modelo nunca vio ni para entrenar ni para ajustar el entrenamiento). Fuente: `reports/metrics.json`.

| Métrica | Valor |
|---|---|
| Accuracy | **80.04%** |
| F1 macro | **79.71%** |
| Pérdida (loss) | 0.561 |

| Clase | Precisión | Recall | F1 | Imágenes |
|---|---|---|---|---|
| Margarita (`daisy`) | 85.0% | 71.6% | 77.7% | 95 |
| Diente de león (`dandelion`) | 87.7% | 79.3% | 83.3% | 135 |
| Rosa (`roses`) | 75.0% | 71.9% | 73.4% | 96 |
| Girasol (`sunflowers`) | 82.1% | 91.4% | 86.5% | 105 |
| Tulipán (`tulips`) | 72.1% | 84.2% | 77.7% | 120 |

**Lectura de los resultados**
- El entrenamiento corrió las 40 épocas; los pesos finales son los de la mejor época (la 38, menor `val_loss`). Entrenamiento ≈ 84 % vs validación ≈ 80 %: la brecha es pequeña, por lo que **no hay sobreajuste importante**.
- Las curvas seguían mejorando lentamente al final, así que con más épocas o *transfer learning* habría margen de mejora.
- Las confusiones principales son **rosa ↔ tulipán** (21 rosas predichas como tulipán y 13 tulipanes como rosa), flores de forma y color parecidos.
- La mejor clase es girasol (F1 ≈ 86 %); la más difícil es rosa (F1 ≈ 73 %).

Figuras en `reports/figures/`: `curvas_entrenamiento.png`, `matriz_confusion.png`, `predicciones_12.png`, `data_augmentation.png`, entre otras.

## 5. Cómo ejecutarlo

### Requisitos
Python 3.11 – 3.13 (probado con 3.12 y 3.13). El modelo se entrenó en Google Colab con Python 3.13, TensorFlow 2.21.0 y Keras 3.13.2.

```bash
git clone https://github.com/<TU_USUARIO>/<TU_REPO>.git
cd <TU_REPO>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### A) Solo usar la aplicación (el modelo ya está en `models/`)
```bash
streamlit run app/streamlit_app.py
```
Se abre en `http://localhost:8501`. Sube una imagen JPG/PNG y verás la clase predicha, la confianza y la probabilidad de cada clase.

### B) Reproducir el entrenamiento en Google Colab
1. Abre `notebooks/01_cnn_flores_experimentacion.ipynb` en Colab (botón de arriba).
2. *Entorno de ejecución → Cambiar tipo de entorno → GPU*.
3. *Ejecutar todo*. Al final se genera y descarga `resultados_colab.zip` con `models/` y `reports/`.

### C) Reproducir el entrenamiento desde la terminal
```bash
python -m src.train          # descarga el dataset, entrena y guarda models/flower_cnn.keras
python -m src.evaluate       # métricas y figuras sobre el conjunto de prueba
python -m src.predict ruta/a/una_imagen.jpg
```
Parámetros opcionales: `--epochs N`, `--data-dir RUTA/flower_photos`.

## 6. Limitaciones
- El modelo solo conoce 5 clases: ante una imagen de otra cosa siempre devolverá una de ellas, y **puede hacerlo con una confianza alta** (la probabilidad softmax no es una medida fiable de certeza fuera de las 5 clases). La app avisa solo cuando la confianza es baja.
- Dataset pequeño y resolución 180×180; mejoras posibles: *transfer learning*, clase "otro", búsqueda de hiperparámetros.
