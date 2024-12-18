import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen
image_path = "500.bmp"
image = cv2.imread(image_path)
original_image = image.copy()

# Redimensionar para visualizar más rápido
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image_resized = cv2.resize(image, (width, height))

# Suavizar la imagen
image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)

# Convertir a espacio HSV
hsv_image = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)

# ---- Segmentación de células moradas ---- #
lower_purple = np.array([100, 50, 50])
upper_purple = np.array([145, 205, 205])
mask_purple = cv2.inRange(hsv_image, lower_purple, upper_purple)

# Procesamiento morfológico para limpiar ruido (células moradas)
kernel = np.ones((3, 3), np.uint8)
mask_purple_cleaned = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((2, 2), np.uint8)
mask_purple_cleaned = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)


# ---- Segmentación de otras células (grises) ---- #
lower_gray = np.array([60, 31, 60])
upper_gray = np.array([200, 255, 255])
mask_gray_hsv = cv2.inRange(hsv_image, lower_gray, upper_gray)
mask_gray_cleaned = cv2.bitwise_and(mask_gray_hsv, cv2.bitwise_not(mask_purple))
#mask_gray_cleaned = cv2.morphologyEx(mask_gray_cleaned, cv2.MORPH_CLOSE, kernel)

# ---- Detectar contornos y rellenar máscaras ---- #
output_image = image_resized.copy()

# Contornos grises
contours_gray, _ = cv2.findContours(mask_gray_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours_gray:
    cv2.drawContours(output_image, [contour], -1, (0, 0, 0), 2)  # Contorno azul
    cv2.drawContours(output_image, [contour], -1, (100, 100, 255), -1)  # Relleno gris claro

# Contornos morados
contours_purple, _ = cv2.findContours(mask_purple_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours_purple:
    cv2.drawContours(output_image, [contour], -1, (75, 50, 142), 2)  # Contorno verde
    cv2.drawContours(output_image, [contour], -1, (255, 100, 100), -1)  # Relleno verde claro


# ---- Visualización de resultados simplificados ---- #
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Imagen Original")
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Segmentación Células Moradas")
plt.imshow(mask_purple_cleaned, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Segmentación Otras Células (Grises)")
plt.imshow(mask_gray_cleaned, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Resultado Final")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
# Guardar la imagen final con contornos (opcional)
output_path = "resultados/filtro_color/deteccion_color.png"
cv2.imwrite(output_path, output_image)
print(f"Imagen procesada guardada en {output_path}")
