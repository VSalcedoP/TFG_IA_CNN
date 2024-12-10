

import pandas as pd

# Declaramos las rutas de los dos dataset orginales
dataset_1 = r'E:\Uni\Asignaturas\TFG\0. Dataset\DFUC2021_train\train.csv'
dataset_2 = r'E:\Uni\Asignaturas\TFG\0. Dataset\Dataset_2.csv'

# Declaramos las rutas de los dataset de salida
dataset_1_output = r'E:\Uni\Asignaturas\TFG\0. Dataset\DFUC2021_train\train_final.csv'
dataset_2_output = r'E:\Uni\Asignaturas\TFG\0. Dataset\PartB_DFU_Dataset\Dataset_2_final.csv'


# Cargamos los CSVs
file_1 = pd.read_csv(dataset_1)
file_2 = pd.read_csv(dataset_2)

# La función Classify establece la clase de la imagen
# en función de los valores de "none, infection, ischaemia y both"
# Tal y como se prepara
def Classify(row):
    result=''

    if row['none']==1:
        result='Healthy'
    elif row['infection']==1:
        result='Infection'
    elif row['ischaemia']==1:
        result='Ischaemia'
    elif row['both']==1:
        result='Both'
    return result

# Eliminamos aquellas filas que no disponen de clasificación en el primer dataset
file_1=file_1.dropna()

# Generamos la nueva columna en los archivos
file_1['Class'] = file_1.apply(Classify, axis=1)
file_2['Class'] = file_2.apply(Classify, axis=1)

# Eliminamos las columnas intermedias ahora que ya están los datos clasificados
file_1=file_1.drop(columns=['none','infection','ischaemia','both'])
file_2=file_2.drop(columns=['none','infection','ischaemia','both'])

# Guardamos los archivos
file_1.to_csv(dataset_1_output, index=False)
file_2.to_csv(dataset_2_output, index=False)