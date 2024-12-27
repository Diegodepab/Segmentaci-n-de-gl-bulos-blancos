from PIL import Image
import os

# Ruta de la carpeta donde están las imágenes BMP
input_folder = "TODAS"
output_folder = "procesadas"

# Verificar si la carpeta de salida existe, si no, crearla
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Función para procesar las imágenes
def process_images(input_folder, output_folder, scale_factor=0.2):
    for filename in os.listdir(input_folder):
        if filename.endswith(".bmp"):
            # Ruta completa del archivo
            img_path = os.path.join(input_folder, filename)
            
            # Abrir la imagen
            with Image.open(img_path) as img:
                # Reducir el tamaño al 20%
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                img_resized = img.resize(new_size, Image.ANTIALIAS)
                
                # Guardar la imagen como PNG en la carpeta de salida
                new_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_folder, new_filename)
                
                # Guardar como PNG con compresión para reducir tamaño
                img_resized.save(output_path, format="PNG", optimize=True, quality=85)
                
                print(f"Imagen {filename} procesada y guardada como {new_filename}")

# Ejecutar el procesamiento
process_images(input_folder, output_folder)
