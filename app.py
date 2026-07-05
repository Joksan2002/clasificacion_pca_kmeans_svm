import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Reconocimiento de Dígitos MNIST",
    page_icon="🔢",
    layout="wide"
)

st.title("Clasificador MNIST con Reducción de Dimensionalidad (PCA, K-Means & SVM) - Joksan Zavala - 20201900395")
st.markdown("""
Esta aplicación web utiliza **PCA** para reducir las dimensiones de imágenes de dígitos manuscritos, 
**K-Means** para analizar su agrupación en clústeres y una **Máquina de Vectores de Soporte (SVM)** para predecir el dígito correcto.
""")

# ==========================================
# Carga de Recursos y Modelos
# ==========================================
@st.cache_resource
def cargar_modelos_y_datos():
    ruta_pca = "models/pca_digit_recognizer.pkl"
    ruta_kmeans = "models/kmeans_digit_clusters.pkl"
    ruta_svm = "models/svm_digit_classifier.pkl"
    ruta_metadata = "models/model_metadata.json"
    ruta_csv = "outputs/mnist_resultados_pca.csv"
    
    pca = joblib.load(ruta_pca) if os.path.exists(ruta_pca) else None
    kmeans = joblib.load(ruta_kmeans) if os.path.exists(ruta_kmeans) else None
    svm = joblib.load(ruta_svm) if os.path.exists(ruta_svm) else None
    
    metadata = {}
    if os.path.exists(ruta_metadata):
        with open(ruta_metadata, "r") as f:
            metadata = json.load(f)
            
    df_pca = pd.read_csv(ruta_csv) if os.path.exists(ruta_csv) else None
    
    return pca, kmeans, svm, metadata, df_pca

pca_model, kmeans_model, svm_model, metadata, df_pca = cargar_modelos_y_datos()

# Validar que los modelos se cargaron correctamente
if pca_model is None or svm_model is None or kmeans_model is None:
    st.error("❌ No se encontraron los modelos en la carpeta `models/`. Asegúrate de haber subido los archivos .pkl correctamente.")
    st.stop()

# ==========================================
# Panel Lateral - Configuración e Interacción
# ==========================================
st.sidebar.header("⚙️ Parámetros del Modelo")

# Seleccionar el número de componentes principales (PCA)
max_componentes = metadata.get("n_components_pca", 35)
n_componentes_seleccionados = st.sidebar.slider(
    "Componentes Principales (PCA) a utilizar", 
    min_value=2, 
    max_value=max_componentes, 
    value=max_componentes
)

st.sidebar.markdown("---")
st.sidebar.header("✏️ Probar con una Imagen del Dataset")

# Carga segura desde el archivo de muestra ligera (100 filas)
@st.cache_data
def cargar_imagenes_muestra():
    ruta_muestra = "mnist_sample.csv"
    if os.path.exists(ruta_muestra):
        df = pd.read_csv(ruta_muestra)
        if "label" in df.columns:
            X = df.drop(columns=["label"]).values
            y = df["label"].values
            return X, y
        else:
            return df.values, np.zeros(len(df))
    return None, None

X_raw, y_raw = cargar_imagenes_muestra()

if X_raw is not None:
    # Cambiar dinámicamente de índice (0 al 99) usando datos reales del archivo CSV
    idx_imagen = st.sidebar.slider("Selecciona el índice de la imagen", 0, len(X_raw) - 1, 10)
    imagen_seleccionada = X_raw[idx_imagen]
    etiqueta_real = y_raw[idx_imagen]
else:
    st.sidebar.error("❌ No se encontró el archivo `mnist_sample.csv` en el repositorio.")
    # Generar un trazo básico de respaldo por si el archivo no se ha subido
    imagen_seleccionada = np.zeros(784)
    imagen_seleccionada[14*28:15*28] = 255
    etiqueta_real = "Desconocido"

# ==========================================
# Sección Principal - Métricas de Evaluación
# ==========================================
st.subheader("📊 Rendimiento Histórico del Modelo")
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.metric(label="Precisión del Clasificador SVM (Accuracy)", value=f"{metadata.get('accuracy_score_svm', 0.0)*100:.2f}%")
with col_m2:
    st.metric(label="Componentes Entrenados en PCA", value=f"{metadata.get('n_components_pca', 0)}")
with col_m3:
    st.metric(label="Varianza Explicada Acumulada", value=f"{metadata.get('varianza_explicada', 0.0)*100:.2f}%")

st.markdown("---")

# ==========================================
# Sección Intermedia - Visualización de Datos (PCA & K-Means)
# ==========================================
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("📉 Proyección 2D de los Datos (PCA)")
    if df_pca is not None:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df_pca, x="PC1", y="PC2", hue="label", 
            palette="tab10", alpha=0.6, ax=ax1, legend="full"
        )
        ax1.set_title("Visualización de las Clases Reales (Dígitos 0-9)")
        st.pyplot(fig1)
    else:
        st.info("Sube el archivo `outputs/mnist_resultados_pca.csv` para ver la proyección 2D.")

with col_graph2:
    st.subheader("🎯 Grupos Generados por K-Means")
    if df_pca is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df_pca, x="PC1", y="PC2", hue="cluster", 
            palette="Set1", alpha=0.6, ax=ax2, legend="full"
        )
        ax2.set_title("Agrupación Visual de Clusters del Espacio PCA")
        st.pyplot(fig2)
    else:
        st.info("Sube el archivo `outputs/mnist_resultados_pca.csv` para ver las agrupaciones de K-Means.")

st.markdown("---")

# ==========================================
# Sección de Inferencia - Predicción de la Imagen Seleccionada
# ==========================================
st.subheader("🔮 Inferencia y Clasificación en Tiempo Real")

col_img, col_pred = st.columns([1, 2])

with col_img:
    st.markdown("**Visualización de la Imagen Seleccionada:**")
    matriz_imagen = imagen_seleccionada.reshape(28, 28)
    fig_img, ax_img = plt.subplots(figsize=(3, 3))
    ax_img.imshow(matriz_imagen, cmap="gray")
    ax_img.axis("off")
    st.pyplot(fig_img)
    st.write(f"**Etiqueta Real en Dataset:** `{etiqueta_real}`")

with col_pred:
    st.markdown("**Resultados obtenidos por los Modelos:**")
    
    # 1. Normalizar y preparar la entrada de la imagen (Rango de 0.0 a 1.0)
    if imagen_seleccionada.max() > 1.0:
        imagen_normalizada = imagen_seleccionada / 255.0
    else:
        imagen_normalizada = imagen_seleccionada
        
    imagen_input = imagen_normalizada.reshape(1, -1)
    
    # 2. Aplicar reducción PCA
    imagen_pca = pca_model.transform(imagen_input)
    
    # Ajustar dinámicamente si el usuario redujo el número de componentes en el slider
    imagen_pca_filtrada = imagen_pca[:, :n_componentes_seleccionados]
    
    # Rellenar con ceros las dimensiones faltantes para mantener consistencia con los modelos .pkl
    imagen_pca_completa = np.zeros((1, max_componentes))
    imagen_pca_completa[:, :n_componentes_seleccionados] = imagen_pca_filtrada

    # 3. Predecir con K-Means y SVM
    cluster_predicho = kmeans_model.predict(imagen_pca_completa)[0]
    clase_predicha_svm = svm_model.predict(imagen_pca_completa)[0]
    
    # Mostrar resultados en la interfaz
    st.info(f"🔹 **Clúster Asignado por K-Means:** `Clúster {cluster_predicho}`")
    st.success(f"🎯 **Clase Predicha por SVM (Dígito Reconocido):** `{clase_predicha_svm}`")
    
    # Breve explicación del proceso técnico
    st.markdown(f"""
    **Explicación del Resultado:** La imagen original de 784 dimensiones (28x28 píxeles) extraída del archivo CSV se comprimió usando los componentes principales de PCA. 
    El algoritmo **K-Means** identificó que sus patrones geométricos corresponden al **Clúster {cluster_predicho}** (agrupación no supervisada). 
    Finalmente, la **SVM** determinó de forma supervisada con una frontera de decisión hiperplana que el número dibujado es un **{clase_predicha_svm}**.
    """)
