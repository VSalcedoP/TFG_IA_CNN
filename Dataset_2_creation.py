
import os
import csv
import shutil

# Declaramos las rutas de las cuatro carpetas en las que están las fotos
carpeta_Inf_Neg = r'E:\Uni\Asignaturas\TFG\0. Dataset\PartB_DFU_Dataset\Infection\Aug-Negative'
carpeta_Inf_Pos = r'E:\Uni\Asignaturas\TFG\0. Dataset\PartB_DFU_Dataset\Infection\Aug-Positive'
carpeta_Isq_Neg = r'E:\Uni\Asignaturas\TFG\0. Dataset\PartB_DFU_Dataset\Ischaemia\Aug-Negative'
carpeta_Isq_Pos = r'E:\Uni\Asignaturas\TFG\0. Dataset\PartB_DFU_Dataset\Ischaemia\Aug-Positive'

# Declaramos la ruta donde se almacenaran las nuevas fotos del dataset 2
nueva_carpeta_dataset2 = r'E:\Uni\Asignaturas\TFG\0. Dataset\PartB_DFU_Dataset\Images'
imagenes_repetidas = []
# Obtenemos el listado de fotos de cada una de ellas
fotos_Inf_Neg = os.listdir(carpeta_Inf_Neg)
fotos_Inf_Pos = os.listdir(carpeta_Inf_Pos)
fotos_Isq_Neg = os.listdir(carpeta_Isq_Neg)
fotos_Isq_Pos = os.listdir(carpeta_Isq_Pos)

# Guardamos en un csv que tendrá la siguiente estructura:
# image,none,infection,ischaemia,both
# Obteniendo así la misma estructura que el otro dataset. 

# Unifico las fotos y después elimino los nombres duplicados
lista_fotos = []
lista_fotos.extend(fotos_Inf_Neg)
lista_fotos.extend(fotos_Inf_Pos)
lista_fotos.extend(fotos_Isq_Neg)
lista_fotos.extend(fotos_Isq_Pos)

fotos_unicas=list(set(lista_fotos))

# Declaramos el nombre del archivo de salida
salida_csv = r'E:\Uni\Asignaturas\TFG\0. Dataset\Dataset_2.csv'

# Creamos el directorio para las nuevas imagenes del dataset 2
if not os.path.exists(nueva_carpeta_dataset2):
    os.makedirs(nueva_carpeta_dataset2)

# Escribe los nombres de archivos en el CSV
with open(salida_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'none', 'infection', 'ischaemia', 'both'])
    for nombre in fotos_unicas:
        isq = 0
        inf = 0
        none = 0
        both = 0

        isq=int(nombre in fotos_Isq_Pos)
        inf=int(nombre in fotos_Inf_Pos)
        none=int((nombre in fotos_Isq_Neg)&(nombre in fotos_Inf_Neg))
        both=int((nombre in fotos_Isq_Pos)&(nombre in fotos_Inf_Pos))
        if none==1:
            isq=0
            inf=0
        if both==1:
            isq=0
            inf=0
        if isq+inf+none+both==1:
            writer.writerow([nombre,none,inf,isq,both])

        if isq==1:
            origen= carpeta_Isq_Pos + '\\'+nombre
        if inf==1:
            origen= carpeta_Inf_Pos + '\\'+nombre
        if none==1:
            origen= carpeta_Isq_Neg + '\\'+nombre
        if both==1:
            origen= carpeta_Isq_Pos + '\\'+nombre


        shutil.copy(origen, os.path.join(nueva_carpeta_dataset2, os.path.basename(origen)))

