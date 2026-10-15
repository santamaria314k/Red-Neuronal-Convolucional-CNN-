"""Aplicación Streamlit: sube una foto de una flor y obtén la clase predicha + confianza.

Ejecutar desde la raíz del repo:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src import config
from src.predict import load_artifacts, predict_image

st.set_page_config(page_title="Clasificador de flores (CNN)", page_icon="🌼")


@st.cache_resource(show_spinner="Cargando modelo...")
def get_model():
    return load_artifacts()


def es(name: str) -> str:
    return config.CLASS_LABELS_ES.get(name, name)


st.title("🌼 Clasificador de flores con CNN")
st.write("Sube una foto de una flor y la red neuronal convolucional indicará de qué tipo es.")

if not config.MODEL_PATH.exists() or not config.CLASS_NAMES_PATH.exists():
    st.error("No se encontró el modelo entrenado en `models/`. "
             "Ejecuta primero el notebook o `python -m src.train`.")
    st.stop()

model, class_names = get_model()

with st.sidebar:
    st.header("Información del modelo")
    st.write("Clases reconocidas:")
    for c in class_names:
        st.write(f"- {es(c)}")
    if config.METRICS_PATH.exists():
        m = json.loads(config.METRICS_PATH.read_text())
        st.metric("Accuracy en test", f"{m['test_accuracy']*100:.1f}%")
        st.metric("F1 macro en test", f"{m['macro_f1']*100:.1f}%")

file = st.file_uploader("Imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if file is not None:
    try:
        image = Image.open(file)
        image.load()
    except Exception:
        st.error("No se pudo leer la imagen. Prueba con otro archivo JPG o PNG.")
        st.stop()

    col1, col2 = st.columns(2)
    col1.image(image, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Analizando..."):
        res = predict_image(model, class_names, image)

    col2.subheader(f"Predicción: {es(res['class'])}")
    col2.metric("Confianza", f"{res['confidence']*100:.2f}%")
    col2.progress(min(max(res["confidence"], 0.0), 1.0))
    if res["confidence"] < config.CONFIDENCE_THRESHOLD:
        col2.warning("Confianza baja: la imagen podría no ser una de las 5 clases "
                     "o no estar bien encuadrada.")

    st.subheader("Probabilidad por clase")
    df = (pd.DataFrame({"Clase": [es(c) for c in res["probabilities"]],
                        "Probabilidad (%)": [p * 100 for p in res["probabilities"].values()]})
          .sort_values("Probabilidad (%)", ascending=False).set_index("Clase"))
    st.bar_chart(df)
    st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
else:
    st.info("Esperando una imagen...")
