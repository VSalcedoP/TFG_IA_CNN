from keras.api import Model
import tensorflow as tf
from keras.api.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.api.callbacks import History
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

# Cargar los modelos
def cargar_modelo(ruta_modelo):
    modelo = load_model(ruta_modelo)
    return modelo

# Cargar las imagenes y etiquetas
def cargar_datos(ruta_imagenes, ruta_etiquetas):
    # Cargamos las etiquetas y las imagenes
    etiquetas_csv = pd.read_csv(ruta_etiquetas)
    imagenes_array = []
    etiquetas = []
    
    for index, row in etiquetas_csv.iterrows():
        ruta_img = ruta_imagenes + row['image']+'.npy'
        imagen = np.load(ruta_img)
        imagenes_array.append(imagen)
        etiquetas.append(row['Class'])
        
    imagenes_array = np.array(imagenes_array)
    etiquetas = np.array(etiquetas)
    
    # Convertir las etiquetas en numeros
    label_encoder = LabelEncoder()
    etiquetas = label_encoder.fit_transform(etiquetas)
    
    # Dividir en entrenamiento y validacion
    X_entrenamiento, X_validacion, Y_entrenamiento, Y_validacion = train_test_split(imagenes_array, etiquetas, test_size=0.2, random_state=42)
    
    return X_entrenamiento, X_validacion, Y_entrenamiento, Y_validacion, label_encoder

# Evaluacion numerica del modelo
def Evaluacion_numerica_del_modelo(modelo, X_validacion, Y_validacion, label_encoder):
    # Evaluamos el modelo con los datos de test
    test_loss, test_accuracy = modelo.evaluate(X_validacion, Y_validacion)

    print(f'Precision obtenida con los datos de test: {test_accuracy*100:.2f}%')

    # Resultados predecidos
    y_pred = modelo.predict(X_validacion)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Matriz de confusion
    cm = confusion_matrix(Y_validacion, y_pred_classes)
    print("Matriz de confusion:")
    print(cm)

    # Precision, recall y f1-score
    print("Informe de clasificacion:")
    print(classification_report(Y_validacion, y_pred_classes, target_names=label_encoder.classes_))

# Evaluacion grafica del modelo
def Evaluacion_grafica(hist):
    # Precision y la perdida
    plt.figure(figsize=(12, 6))

    # Precision
    plt.subplot(1, 2, 1)
    plt.plot(hist['accuracy'], label='Entrenamiento')
    plt.plot(hist['val_accuracy'], label='Validacion')
    plt.title('Precision del Modelo')
    plt.xlabel('epocas')
    plt.ylabel('Precision')
    plt.legend()

    # Perdida
    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'], label='Entrenamiento')
    plt.plot(hist['val_loss'], label='Validacion')
    plt.title('Perdida del Modelo')
    plt.xlabel('epocas')
    plt.ylabel('Perdida')
    plt.legend()

    plt.show()

# Establecemos un parametro para decidir que dataset queremos procesar
Dataset_Pr=3

# Ruta de los modelos
ruta_modelo_base = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\\modelo_dataset_'+str(Dataset_Pr)+'.h5'
ruta_modelo_ft = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\\modelo_dataset_'+str(Dataset_Pr)+'_FT.h5'

# Ruta de imagenes y etiquetas
ruta_imagenes = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\Dataset_'+str(Dataset_Pr)+'\Pre-processed_Images\\'
ruta_etiquetas = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\Dataset_'+str(Dataset_Pr)+'\Dataset_'+str(Dataset_Pr)+'_final.csv'

# Ruta historiales
ruta_modelo_base_historial = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\\modelo_dataset_'+str(Dataset_Pr)+'_historial.json'
ruta_modelo_ft_historial = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\\modelo_dataset_'+str(Dataset_Pr)+'_historial_FT.json'

# Cargamos los datos
X_entrenamiento, X_validacion, Y_entrenamiento, Y_validacion, label_encoder = cargar_datos(ruta_imagenes, ruta_etiquetas)

# Cargamos los modelos
modelo_base: Model
modelo_ft: Model
modelo_base = cargar_modelo(ruta_modelo_base)
modelo_ft = cargar_modelo(ruta_modelo_ft)

# Cargamos los historiales de los modelos
with open(ruta_modelo_base_historial, 'r') as f:
    historial_base= json.load(f)

with open(ruta_modelo_ft_historial, 'r') as f:
    historial_FT= json.load(f)

# Evaluacion del modelo base
print("Evaluacion del modelo base:")
Evaluacion_numerica_del_modelo(modelo_base, X_validacion, Y_validacion, label_encoder)

Evaluacion_grafica(historial_base)

# Evaluacion del modelo Fine Tuning
print("\nEvaluacion del modelo Fine Tuning:")
Evaluacion_numerica_del_modelo(modelo_ft, X_validacion, Y_validacion, label_encoder)

Evaluacion_grafica(historial_FT)
