import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
# Cabecera con Logotipo y Título
col_logo, col_titulo = st.columns([1, 4])  # Proporción de ancho

with col_logo:
    try:
        st.image("Logo_CT_letras.png", width=120)
    except Exception:
        pass

with col_titulo:
    st.title("🌱 CivilTurf GrassCoverCalc")
    st.write("Aplicación para el cálculo de cobertura vegetal en césped")
st.set_page_config(page_title="CivilTurfGrassCover - Cobertura Vegetal Multi-Imagen", layout="wide")

#st.title("🌱 Análisis de Cobertura Vegetal por imágen en césped")
st.write("Carga una o varias imágenes para procesar de forma individual o en lote mediante las relaciones de color R/G, B/G y Exceso de Verde (ExG).")

# Botón para limpiar fotos y realizar una nueva medición rápidamente
col_info, col_reset = st.columns([3, 1])
with col_reset:
    if st.button("🔄 Limpiar y analizar nuevo lote", use_container_width=True):
        st.session_state["uploader_key"] += 1
        st.rerun()

# Configuración de parámetros CT en la barra lateral
st.sidebar.header("Parámetros TurfGrassCover")
p1 = st.sidebar.slider("P1 (Límite R/G)", min_value=0.50, max_value=1.50, value=0.95, step=0.01)
p2 = st.sidebar.slider("P2 (Límite B/G)", min_value=0.50, max_value=1.50, value=0.95, step=0.01)
p3 = st.sidebar.slider("P3 (Mínimo 2G - R - B)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

# Función auxiliar para sobreimprimir la marca de agua con el porcentaje en la imagen
def agregar_etiqueta_cobertura(imagen_pil, porcentaje):
    img = imagen_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    texto = f"Cobertura: {porcentaje:.2f}%"
    
    # Adaptar el tamaño de la fuente según la resolución de la imagen
    ancho, alto = img.size
    tamanio_fuente = max(18, int(alto * 0.035))
    
    try:
        fuente = ImageFont.load_default(size=tamanio_fuente)
    except Exception:
        fuente = ImageFont.load_default()

    # Coordenadas y recuadro de fondo oscuro para maximizar la legibilidad
    pos_x, pos_y = 15, 15
    bbox = draw.textbbox((pos_x, pos_y), texto, font=fuente)
    
    # Dibujar cuadro de fondo negro semitransparente/sólido
    draw.rectangle(
        [bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6], 
        fill=(0, 0, 0)
    )
    # Dibujar texto en color blanco
    draw.text((pos_x, pos_y), texto, fill=(255, 255, 255), font=fuente)
    
    return img

# Carga de imágenes con clave dinámica de sesión
uploaded_files = st.file_uploader(
    "Selecciona una o varias imágenes (JPG, PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}"
)

if uploaded_files:
    resultados = []
    imagenes_procesadas = []

    # Bucle de procesamiento numerando cada imagen subida
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        nombre_archivo_orig = uploaded_file.name
        if nombre_archivo_orig.lower() in ["image.jpg", "image.jpeg", "image.png"]:
            nombre_display = f"Imagen_{idx}.jpg"
        else:
            nombre_display = f"Foto_{idx}_{nombre_archivo_orig}"

        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image, dtype=np.float32)

        # Extraer canales R, G y B
        R = img_np[:, :, 0]
        G = img_np[:, :, 1]
        B = img_np[:, :, 2]

        # Evaluación de las 3 condiciones Canopeo
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
        img_overlay_np = np.array(image).copy()
        img_overlay_np[mascara_binaria == 0] = [255, 0, 0]
        img_overlay_pil = Image.fromarray(img_overlay_np)

        # Convertir máscara a PIL para dibujo
        mascara_pil = Image.fromarray(mascara_binaria * 255)

        # --- ANOTAR EL PORCENTAJE SOBRE CADA UNA DE LAS 3 IMÁGENES ---
        img_original_anotada = agregar_etiqueta_cobertura(image, cobertura_porcentaje)
        mascara_anotada = agregar_etiqueta_cobertura(mascara_pil, cobertura_porcentaje)
        overlay_anotado = agregar_etiqueta_cobertura(img_overlay_pil, cobertura_porcentaje)

        # Guardar resultados numéricos
        resultados.append({
            "Archivo": nombre_display,
            "Cobertura Vegetal (%)": round(cobertura_porcentaje, 2),
            "Píxeles Vegetación": int(pixeles_vegetacion),
            "Píxeles Totales": int(pixeles_totales)
        })

        # Guardar las imágenes anotadas
        imagenes_procesadas.append({
            "nombre": nombre_display,
            "image": img_original_anotada,
            "mascara": mascara_anotada,
            "overlay": overlay_anotado,
            "cobertura": cobertura_porcentaje
        })

    # Convertir resultados a DataFrame
    df_resultados = pd.DataFrame(resultados)
    promedio_cobertura = df_resultados["Cobertura Vegetal (%)"].mean()

    # --- RESUMEN GLOBAL ---
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
    
    # --- INSPECTOR INDIVIDUAL ---
    st.subheader("🔍 Inspección Detallada")

    nombres_archivos = [img["nombre"] for img in imagenes_procesadas]
    imagen_seleccionada_nombre = st.selectbox("Selecciona una fotografía para revisar sus máscaras:", nombres_archivos)

    datos_img = next(item for item in imagenes_procesadas if item["nombre"] == imagen_seleccionada_nombre)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Fotografía Original")
        st.image(datos_img["image"], use_container_width=True)
    with col2:
        st.subheader("Máscara Binaria")
        st.image(datos_img["mascara"], use_container_width=True)
    with col3:
        st.subheader("Superposición (Suelo en Rojo)")
        st.image(datos_img["overlay"], use_container_width=True)