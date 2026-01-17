import streamlit as st
import cv2
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="OCR Inventario PRO", layout="wide")
st.title("📦 OCR Inventario PRO (Python OCR PRO)")

# ======================
# CONFIG TESSERACT
# ======================
pytesseract.pytesseract.tesseract_cmd = "tesseract"

# ======================
# SUBIR IMAGEN
# ======================
uploaded = st.file_uploader(
    "📷 Carga una imagen de sticker",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)
    img = np.array(image)

    st.image(img, caption="Imagen original", use_column_width=True)

    # ======================
    # PREPROCESADO PRO
    # ======================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    # ======================
    # AUTODETECCIÓN DE LÍNEAS
    # ======================
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    h_img = img.shape[0]

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if h > 15 and w > 50:
            lines.append((y, x, w, h))

    lines = sorted(lines)

    resultados = []
    preview = img.copy()

    # ======================
    # OCR POR LÍNEA
    # ======================
    for i, (y,x,w,h) in enumerate(lines):
        crop = gray[y:y+h, x:x+w]

        text = pytesseract.image_to_string(
            crop,
            lang="spa+eng",
            config="--psm 7"
        ).strip()

        if text:
            resultados.append(text)
            cv2.rectangle(preview, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(preview, str(i+1), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ======================
    # RESULTADOS
    # ======================
    st.subheader("🔍 Líneas detectadas")
    st.image(preview, use_column_width=True)

    if resultados:
        df = pd.DataFrame({
            "#": range(1, len(resultados)+1),
            "Descripción": resultados
        })

        st.subheader("📋 Resultado OCR")
        st.data_editor(df, use_container_width=True)

        # ======================
        # EXPORTAR EXCEL
        # ======================
        excel = df.to_excel(index=False, engine="openpyxl")
        st.download_button(
            "⬇️ Descargar Excel",
            data=excel,
            file_name="inventario_ocr_pro.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("⚠️ No se detectó texto")
