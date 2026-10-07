import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(page_title="CivilTurfGrassCover - Cobertura Vegetal Multi-Imagen", layout="wide")

st.title("🌱 Análisis de Cobertura Vegetal (Algoritmo CT - Multi-Imagen)")
st.write("Carga una o varias imágenes para procesar de forma individual o en lote mediante las relaciones de color R/G, B/G y Exceso de Verde (ExG).")

# Configuración de parámetros CT en la barra lateral
st.sidebar.header("Parámetros TurfGrassCover")
p1 = st.sidebar.slider("P1 (Límite R/G)", min_value=0.50, max_value=1.50, value=0.95, step=0.01)
p2 = st.sidebar.slider("P2 (Límite B/G)", min_value=0.50, max_value=1.50, value=0.95, step=0.01)
p3 = st.sidebar.slider("P3 (Mínimo 2G - R - B)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

# Carga de múltiples imágenes simultáneamente
uploaded_files = st.file_uploader(
    "Selecciona una o varias imágenes (JPG, PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    resultados = []
    imagenes_procesadas = []

    # Bucle de procesamiento para cada imagen subida
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image, dtype=np.float32)

        # Extraer canales R, G y B
        R = img_np[:, :, 0]
        G = img_np[:, :, 1]
        B = img_np[:, :, 2]

        # Evaluación de las 3 condiciones CivilTurfGrassCover
        cond1 = np.where(G > 0, (R / G) < p1, False)
        cond2 = np.where(G > 0, (B / G) < p2, False)
        cond3 = (2 * G - R - B) > p3

        # Máscara binaria resultante
        mascara_binaria = (cond1 & cond2 & cond3).astype(np.uint8)

        # Cálculos de superficie
        pixeles_vegetacion = np.sum(mascara_binaria)
        pixeles_totales = mascara_binaria.size
        cobertura_porcentaje = (pixeles_vegetacion / pixeles_totales) * 100.0

        # Imagen de superposición (suelo en rojo)
        img_overlay = np.array(image).copy()
        img_overlay[mascara_binaria == 0] = [255, 0, 0]

        # Guardar resultados numéricos
        resultados.append({
            "Archivo": uploaded_file.name,
            "Cobertura Vegetal (%)": round(cobertura_porcentaje, 2),
            "Píxeles Vegetación": int(pixeles_vegetacion),
            "Píxeles Totales": int(pixeles_totales)
        })

        # Guardar matrices para la visualización gráfica
        imagenes_procesadas.append({
            "nombre": uploaded_file.name,
            "image": image,
            "mascara": mascara_binaria,
            "overlay": img_overlay,
            "cobertura": cobertura_porcentaje
        })

    # Convertir resultados a DataFrame de Pandas
    df_resultados = pd.DataFrame(resultados)
    promedio_cobertura = df_resultados["Cobertura Vegetal (%)"].mean()

    # --- SECCIÓN DE RESUMEN GLOBAL ---
    st.subheader(f"📊 Resumen de Resultados ({len(uploaded_files)} {'imagen' if len(uploaded_files) == 1 else 'imágenes'})")
    
    col_res1, col_res2 = st.columns([1, 3])
    with col_res1:
        st.metric("Cobertura Media Global", f"{promedio_cobertura:.2f} %")
        
        # Botón para descargar los datos procesados a CSV
        csv = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Descargar Informe CSV",
            data=csv,
            file_name="informe_cobertura_canopeo.csv",
            mime="text/csv",
        )
    with col_res2:
        st.dataframe(df_resultados, use_container_width=True)

    st.markdown("---")
    
    # --- INSPECTOR INDIVIDUAL DE IMÁGENES ---
    st.subheader("🔍 Inspección Detallada")

    nombres_archivos = [img["nombre"] for img in imagenes_procesadas]
    imagen_seleccionada_nombre = st.selectbox("Selecciona una fotografía para revisar sus máscaras:", nombres_archivos)

    # Obtener los datos de la imagen seleccionada en el dropdown
    datos_img = next(item for item in imagenes_procesadas if item["nombre"] == imagen_seleccionada_nombre)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Fotografía Original")
        st.image(datos_img["image"], use_container_width=True)
    with col2:
        st.subheader(f"Máscara Binaria ({datos_img['cobertura']:.2f}%)")
        st.image(datos_img["mascara"] * 255, use_container_width=True)
    with col3:
        st.subheader("Superposición (Suelo en Rojo)")
        st.image(datos_img["overlay"], use_container_width=True)