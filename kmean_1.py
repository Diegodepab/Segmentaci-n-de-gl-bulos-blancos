# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:22:46 2024

@author: DiegoDePablo
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen
image_path = "500.bmp"  # Reemplaza con tu imagen
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redimensionar (opcional)
scale_percent = 50  # Escalar para procesar más rápido
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image_resized = cv2.resize(image_rgb, dim)

# Aplanar la imagen (reshape) para K-Means
pixel_values = image_resized.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Configurar colores personalizados para los clusters (colores pastel más agradables)
custom_colors = np.array([
    [65, 176, 110],  #  rgb(65, 176, 110)
    [33, 53, 85],  # rgb(33, 53, 85)
    [62, 88, 121],  # rgb(62, 88, 121)
    [245, 239, 231]   # rgb(245, 239, 231)
])
k = custom_colors.shape[0]  # Número de clusters basado en los colores

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
plt.subplot(2, 1, 1)  # Dividimos en dos filas
plt.title("Imagen Original")
plt.imshow(image_resized)
plt.axis("off")

# Imagen Segmentada
plt.subplot(2, 1, 2)  # Imagen segmentada en la fila inferior
plt.title("Imagen Segmentada")
plt.imshow(segmented_image)
plt.axis("off")

# Crear leyenda debajo de la imagen segmentada
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color / 255.0, markersize=10)
                  for color in custom_colors]
plt.legend(legend_handles, ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"],
           loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)

plt.tight_layout()
plt.show()
