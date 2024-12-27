import cv2
import os

# Variables globales
points = []  # Lista para almacenar los puntos seleccionados
labels = []  # Lista para almacenar las etiquetas
radii = []   # Lista para almacenar el tamaño de la célula (radio)
current_label = 1  # Etiqueta inicial, puede ser 1 (glóbulos blancos), 2 (otros grupos), etc.
img = None  # Variable global para la imagen actual

def draw_circle(event, x, y, flags, param):
    """Función de callback para manejar los clics del mouse."""
    global points, labels, radii, current_label
    if event == cv2.EVENT_LBUTTONDOWN:  # Cuando se hace clic izquierdo
        points.append((x, y))
        labels.append(current_label)  # Etiqueta actual
        radius = input(f"Introduce el tamaño de la célula (radio) para el punto en {x}, {y}: ")
        radii.append(float(radius))  # Guardar el radio proporcionado por el usuario
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Dibuja un pequeño círculo en la imagen
        cv2.imshow('image', img)

# Ruta de la carpeta de imágenes
input_folder = 'procesadas'

# Listar todas las imágenes en formato PNG
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Procesar cada imagen
for image_file in image_files:
    img_path = os.path.join(input_folder, image_file)
    img = cv2.imread(img_path)  # Cargar la imagen
    
    if img is None:
        print(f"No se pudo cargar la imagen {image_file}")
        continue
    
    # Mostrar la imagen
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', draw_circle)  # Asignar la función de callback para capturar los clics

    while True:
        key = cv2.waitKey(0) & 0xFF  # Esperar una tecla
        if key == ord('n'):  # Presiona 'n' para ir a la siguiente imagen
            break
        elif key == ord('1'):  # Cambiar la etiqueta a 1 (por ejemplo, glóbulos blancos)
            current_label = 1
            print("Etiqueta cambiada a 1 (glóbulos blancos)")
        elif key == ord('2'):  # Cambiar la etiqueta a 2 (otro tipo de célula)
            current_label = 2
            print("Etiqueta cambiada a 2 (otro grupo de interés)")
        elif key == ord('q'):  # Presiona 'q' para salir del programa
            exit()
    
    # Guardar puntos, radios y etiquetas al terminar con la imagen
    if points:
        label_file_path = f'etiquetas_{os.path.splitext(image_file)[0]}.txt'
        with open(label_file_path, 'w') as f:
            for point, label, radius in zip(points, labels, radii):
                f.write(f'{point[0]},{point[1]},{label},{radius}\n')
        print(f"Puntos etiquetados guardados en {label_file_path}")

    # Limpiar puntos, etiquetas y radios para la siguiente imagen
    points.clear()
    labels.clear()
    radii.clear()

cv2.destroyAllWindows()
