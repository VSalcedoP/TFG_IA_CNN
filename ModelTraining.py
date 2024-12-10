import os
import tensorflow as tf
from keras.api.applications import InceptionV3
from keras.api import layers, models, Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.api.callbacks import History, EarlyStopping
import json

# Establecemos un parametro para decidir que dataset queremos procesar
Dataset_Pr=3

# Carpeta de imagenes
ruta_imagenes = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\Dataset_'+str(Dataset_Pr)+'\\Pre-processed_Images\\'


# Csv con etiquetas
ruta_etiquetas = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\Dataset_'+str(Dataset_Pr)+'\\Dataset_'+str(Dataset_Pr)+'_final.csv'

# Entrenamiento
X_entrenamiento = []
Y_entrenamiento = []

# Validacion
X_validacion = []
Y_validacion = []

# Etiquetas originales
label_encoder = LabelEncoder()

# Ruta modelo
ruta_modelo = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\\'

# Modelo base
modelo_inicial: Model

def Dividir_Dataset():
    # Separamos las etiquetas en entrenamiento (80%) y validacion (20%)
    global X_entrenamiento, X_validacion, Y_entrenamiento, Y_validacion

    # Cargamos el csv con las etiquetas
    etiquetas_csv = pd.read_csv(ruta_etiquetas)

    # Cargamos los arrays Numpy creados anteriormente y los asociamos con sus etiquetas
    etiquetas = []
    imagenes_array = []

    for index, row in etiquetas_csv.iterrows():
        ruta_img = ruta_imagenes + row['image']+'.npy'
        imagen = np.load(ruta_img)
        imagenes_array.append(imagen)
        etiquetas.append(row['Class'])

    imagenes_array = np.array(imagenes_array)
    etiquetas = np.array(etiquetas)

    # Convertimos las etiquetas de las clases a valores numericos por necesidades del modelo
 
    etiquetas = label_encoder.fit_transform(etiquetas)

    # Separamos los datos en conjuntos de entranamiento y test (80% - 20%)
    X_entrenamiento, X_validacion, Y_entrenamiento, Y_validacion = train_test_split(imagenes_array, etiquetas, test_size=0.2, random_state=42)

def Crear_modelo():
    # Creamos el modelo partiendo del modelo base de InceptionV3
    global modelo_inicial
    # Cargamos InceptionV3 sin la capa final
    modelo_inicial = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # Congelamos las capas base del modelo preentrenado
    modelo_inicial.trainable = False

    # Creamos el modelo completo
    modelo = models.Sequential()

    # Agregamos el modelo base de InceptionV3 (sin la capa final)
    modelo.add(modelo_inicial)

    # Agregar capas adicionales
    modelo.add(layers.GlobalAveragePooling2D())  # Reduccion dimensional
    modelo.add(layers.Dense(256, activation='relu'))
    modelo.add(layers.Dropout(0.5))  # Dropout para prevenir sobreajuste
    modelo.add(layers.Dense(4, activation='softmax'))  # Capa de clasificacion para 4 clases

    # Compilamos el modelo con un optimizador Adam
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Devolvemos el modelo creado
    return modelo
def Entrenamiento_modelo (modelo: Model, epocas):
    # Entrenamos el modelo con los datos de los arraya
    global X_entrenamiento, X_validacion, Y_entrenamiento, Y_validacion
    return modelo.fit(X_entrenamiento, Y_entrenamiento, epochs=epocas, batch_size=32, validation_data=(X_validacion, Y_validacion))

def Fine_Tuning(capas: int, modelo: Model, epocas: int, lr):
    # El modelo original no es especifico para imagenes medicas
    # Por lo tanto podemos descongelar bastantes capas aunque
    # teniendo en cuenta que podria llevar a sobreajustar el modelo.
    # La tasa de aprendizaje se establece mucho mas pequeña que la original (0.001)
    # Ya que las capas pre-entrenadas ya diosponen de caracteristicas útiles y no deseamos
    # modificarlas en exceso.
    global X_entrenamiento, X_validacion, Y_entrenamiento, Y_validacion, modelo_inicial, learning
    # Descongelamos las capas que se decida
    modelo_inicial.trainable=True
    for layer in modelo_inicial.layers[:-capas]:
        layer.trainable = False

    # Recompilar el modelo (importante despues de cambiar el atributo `trainable`)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    return modelo.fit(X_entrenamiento, Y_entrenamiento, epochs=epocas, batch_size=32, validation_data=(X_validacion, Y_validacion),callbacks=[early_stopping])
def guardar_historial_json(historial, ruta_archivo):
     # Extraemos el historial como un diccionario
    historial_dict = historial.history

    # Guardamos el diccionario en un archivo JSON
    with open(ruta_archivo, 'w') as archivo_json:
        json.dump(historial_dict, archivo_json)

# Proceso de trabajo
Dividir_Dataset()
Modelo = Crear_modelo()

Epocas_modelo_inicial=10
Epocas_modelo_FT=15

# learning_inicial=0.0001
learning_FT=0.0001

# Capas a descongelar
capas=4

Resultado_entrenamiento = Entrenamiento_modelo (Modelo, Epocas_modelo_inicial)
guardar_historial_json(Resultado_entrenamiento,ruta_modelo+'modelo_dataset_'+str(Dataset_Pr)+'_historial.json')

Modelo.save(ruta_modelo+'modelo_dataset_'+str(Dataset_Pr)+'.h5')

#Fine Tuning
Resultado_entrenamiento_FT=Fine_Tuning(capas, Modelo, Epocas_modelo_FT, learning_FT)

Modelo.save(ruta_modelo+'modelo_dataset_'+str(Dataset_Pr)+'_FT.h5')
guardar_historial_json(Resultado_entrenamiento_FT,ruta_modelo+'modelo_dataset_'+str(Dataset_Pr)+'_historial_FT.json')
