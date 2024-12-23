# Segmentación de Imágenes de Células Plasmáticas en Tejido Neoplásico
## Dataset

Las imágenes usadas en este trabajo corresponden a Gupta, R., & Gupta, A. (2019). MiMM_SBILab Dataset: Microscopic Images of Multiple Myeloma [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/tcia.2019.pnn6aypl
](https://doi.org/10.7937/tcia.2019.pnn6aypl
)
Las imágenes microscópicas se obtuvieron de portaobjetos de aspirado de médula ósea de pacientes con diagnóstico de mieloma múltiple según las pautas estándar. Los portaobjetos se tiñeron con tinción de Jenner-Giemsa. Las imágenes se capturaron con un aumento de 1000x utilizando un microscopio Nikon Eclipse-200 equipado con una cámara digital. Las imágenes se capturaron en formato BMP sin procesar con un tamaño de 2560x1920 píxeles. En total, este conjunto de datos consta de 85 imágenes.

## Descripción

Este proyecto tiene como objetivo segmentar imágenes de células plasmáticas en tejido neoplásico, utilizando diversas técnicas de segmentación, tanto supervisadas como no supervisadas. Se implementan métodos de K-Means, Mixturas de Gaussianas (GMM), y Segmentación Semántica, con un enfoque en mejorar la precisión y efectividad del proceso de segmentación en imágenes médicas.

## Estructura del Proyecto

El proyecto está organizado en los siguientes directorios y archivos:
```
/Segmentacion_Celulas_Plasmaticas
│
├── /imagenes                      # Imágenes sin subir debido a su exagerado peso
│
├── /scripts                       # Scripts para procesamiento y segmentación de imágenes.
│   ├── segmentar_por_RGB.py       # Script que no aplica aprendizaje computacional, simplemente segmenta por tonalidades RGB
│   ├── agrupar_1_imagen_kmeans.py # Usar kmean para agrupar teniendo en cuenta solamente los patrones de 1 imagen
│   ├── Kmeans_entrenado.py        # Entrenar Kmean con imagenes de entrenamiento, pero simplificando sus patrones.
│   ├── Mixturas_Gaussianas.Rmd    # En R, segmentar usando Mixturas Gaussianas
│   ├──SegmentacionSemantica.ipynb # En un jupyter notebook, aplicar aprendizaje supervisado
├── /reporte                       # Documentación relacionada con el proyecto.
│   └── informe.pdf                # Informe detallado sobre los métodos y resultados.
│
└── README.md                      # Este archivo.
```


## Requisitos

Para ejecutar este proyecto, necesitas tener instalados los siguientes paquetes:

- Python 3.x
- Bibliotecas de Python:
  - `numpy`
  - `scikit-learn`
  - `opencv-python`
  - `matplotlib`
  - `mclust` (para Mixturas de Gaussianas)
  - `tensorflow` (para Mask R-CNN)
  - `keras`
  - `pillow`
  - `imageio`
  - `pandas`
  
## Descripción de los Métodos
- **Segmentación de imágenes basada en color:** Al contrario de las demás técnicas y modelos usados, este es un script que no aplica aprendizaje computacional, será usado como referencia para contrastar contra los demás métodos. 
- **K-Means:** es una técnica de clustering no supervisada utilizada para detectar patrones en los píxeles de las imágenes. En este trabajo, se utiliza en dos fases: una sin entrenamiento y otra con un entrenamiento adecuado para segmentar las imágenes de células plasmáticas.
- **Mixturas de Gaussianas (GMM):** son un modelo probabilístico que asume que los datos provienen de una mezcla de varias distribuciones gaussianas. En este trabajo, se usa para segmentar las imágenes al modelar la distribución de los píxeles y mejorar la precisión de la segmentación.
- **Segmentación Semántica:** se utilizan modelos preentrenados como el Mask R-CNN para etiquetar y segmentar semánticamente las imágenes. Este enfoque permite identificar de manera precisa las regiones de interés dentro de las imágenes médicas.

