import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Cargar imágenes desde una carpeta
def load_images_from_folder(folder_path, scale_percent=50):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".bmp"):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                # Redimensionar (opcional para acelerar el procesamiento)
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                img = cv2.resize(img, (width, height))
                images.append(img)
    return images

# 2. Extraer píxeles de todas las imágenes
def extract_pixels(images):
    pixel_data = []
    for img in images:
        # Convertir la imagen a RGB y aplanarla
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1, 3))
        pixel_data.append(pixels)
    return np.vstack(pixel_data)  # Combinar todos los píxeles

# 3. Aplicar K-Means a los píxeles combinados
def train_kmeans(pixel_data, k=3):
    print("Entrenando K-Means...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixel_data)
    return kmeans

# 4. Aplicar K-Means a una imagen para segmentación
def segment_image(image, kmeans_model):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = img_rgb.reshape((-1, 3))
    labels = kmeans_model.predict(pixel_values)
    centers = np.uint8(kmeans_model.cluster_centers_)
    
    # Reconstruir la imagen segmentada
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img_rgb.shape)
    return segmented_image

# ---- Main ----
folder_path = "entrenamiento"  
test_image_path = "500.bmp"  # Imagen a testear

# Cargar imágenes del dataset
print("Cargando imágenes del dataset...")
images = load_images_from_folder(folder_path)

# Extraer píxeles de todas las imágenes
print("Extrayendo características de color...")
pixel_data = extract_pixels(images)

# Entrenar K-Means
k = 4  # se vio en pruebas menores el mejor
kmeans_model = train_kmeans(pixel_data, k=k)

# ---- Segmentar una imagen de prueba ----
print("Segmentando imagen de prueba...")
test_image = cv2.imread(test_image_path)
segmented_result = segment_image(test_image, kmeans_model)

# Mostrar el resultado de segmentación
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagen Segmentada (K-Means)")
plt.imshow(segmented_result)
plt.axis("off")
plt.show()

# ---- Cálculo de métricas finales ----
# 1. Inercia del modelo (Within-cluster sum of squares)
print("Inercia del modelo K-Means:", kmeans_model.inertia_)

# 2. Índice de Silueta (usando una muestra si los datos son demasiado grandes)
sample_size = 10000  # Ajusta según tu capacidad de cómputo
if pixel_data.shape[0] > sample_size:
    pixel_sample = pixel_data[np.random.choice(pixel_data.shape[0], sample_size, replace=False)]
    labels_sample = kmeans_model.predict(pixel_sample)
    silhouette_avg = silhouette_score(pixel_sample, labels_sample)
else:
    silhouette_avg = silhouette_score(pixel_data, kmeans_model.labels_)

print("Índice de silueta promedio:", silhouette_avg)
