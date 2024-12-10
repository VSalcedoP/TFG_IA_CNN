import os
import tensorflow as tf
from keras.api.preprocessing import image
from keras.api.applications.inception_v3 import preprocess_input
from keras.api.preprocessing.image import load_img, img_to_array
import shutil
from PIL import Image
import numpy as np

# Establecemos un parámetro para decidir que dataset queremos procesar
Dataset_Pr=2

# Carpeta de imágenes
image_path = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\Dataset_'+str(Dataset_Pr)+'\\Images'

# Carpeta de salida
Output_folder = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\Dataset_'+str(Dataset_Pr)+'\\Pre-processed_Images'

# Definimos el tamaño pedido por inception
InceptionV3_image_size = (299, 299)

# Creamos la carpeta de salida en caso de que no exista
if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)

for archivo in os.listdir(image_path):
    
    # Creamos la dirección completa de la imagen ya que listdir solo devuelve el nombre
    # En la carpeta hay exclusivamente imágenes por lo que no realizo control sobre su contenido
    ruta_imagen_completa=os.path.join(image_path,archivo)

    # Nos aseguramos de que las imágenes esten en formato RGB
    img = Image.open(ruta_imagen_completa)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    #Redimensionamos la imagen
    imagen = img.resize(InceptionV3_image_size)

    # Obtenemos un array a partir de la imágen para poder normalizarlo
    imagen_array = img_to_array(imagen)
    imagen_array = preprocess_input(imagen_array)

    # Guardamos la imagen en la carpeta de salida como array Numpy
    ruta_imagen_salida = os.path.join(Output_folder,archivo)
    np.save(ruta_imagen_salida,imagen_array)
    