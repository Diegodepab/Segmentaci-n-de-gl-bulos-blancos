# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:22:46 2024

@author: DiegoDePablo
"""
################# KMEAN con una imagen sin entrenar ########################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Cargar imagen
image_path = "500.bmp"  
# Número de clusters (K)
k = 4  # Hice una función donde varia la K y el mejor resultado para mostrar claramente los linfocitos, otras celulas y el fondo

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redimensionar (Como indico en clase reducimos para evitar complejidad y gasto computacional excesivo)
scale_percent = 20  # Escalar para procesar más rápido
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image_resized = cv2.resize(image_rgb, dim)

# Aplanar la imagen (reshape) para K-Means
pixel_values = image_resized.reshape((-1, 3))
pixel_values = np.float32(pixel_values)



# Colores personalizados
default_colors = np.array([
    [65, 176, 110],  # rgb(65, 176, 110)
    [33, 53, 85],    # rgb(33, 53, 85)
    [62, 88, 121],   # rgb(62, 88, 121)
    [245, 239, 231], # rgb(245, 239, 231)
    [255, 195, 0],   # rgb(255, 195, 0)
    [230, 74, 25],   # rgb(230, 74, 25)
    [126, 87, 194]   # rgb(126, 87, 194)
])
if k <= len(default_colors):
    custom_colors = default_colors[:k]
else:
    # Generar colores adicionales automáticamente si K > len(default_colors)
    np.random.seed(69)  # Para reproducibilidad
    custom_colors = np.vstack([default_colors, np.random.randint(0, 256, size=(k - len(default_colors), 3))])

# Aplicar K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
_, labels, _ = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Reconstruir imagen segmentada con colores personalizados
custom_colors = np.uint8(custom_colors)
segmented_image = custom_colors[labels.flatten()]
segmented_image = segmented_image.reshape(image_resized.shape)

# Mostrar resultados
plt.figure(figsize=(12, 8))

# Imagen Original
plt.subplot(2, 1, 1)
plt.title("Imagen Original")
plt.imshow(image_resized)
plt.axis("off")

# Imagen Segmentada
plt.subplot(2, 1, 2)
plt.title("Imagen Segmentada")
plt.imshow(segmented_image)
plt.axis("off")

# Crear leyenda basada en los colores generados
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color / 255.0, markersize=10)
    for color in custom_colors
]
legend_labels = [f"Cluster {i + 1}" for i in range(k)]
plt.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=min(5, k), frameon=False)

plt.tight_layout()
plt.show()

# Después de aplicar K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Inercia del modelo K-Means (compactness es similar a la inercia)
print("Inercia del modelo K-Means (sin entrenar):", compactness)

# Índice de Silueta
sample_size = int(len(pixel_values) * 0.1)  # Tomar el 10% de los píxeles
pixel_sample = pixel_values[np.random.choice(pixel_values.shape[0], sample_size, replace=False)]
labels_sample = labels.flatten()[np.random.choice(labels.flatten().shape[0], sample_size, replace=False)]

silhouette_avg = silhouette_score(pixel_sample, labels_sample)
print("Índice de silueta promedio (muestra):", silhouette_avg)

