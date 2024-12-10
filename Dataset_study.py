

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Declaramos las rutas de los dataset
dataset_1 = r'E:\Uni\Asignaturas\TFG\0. Dataset\DFUC2021_train\train_final.csv'
dataset_2 = r'E:\Uni\Asignaturas\TFG\0. Dataset\PartB_DFU_Dataset\Dataset_2_final.csv'
dataset_3 = r'E:\Uni\Asignaturas\TFG\0. Dataset\Work\Dataset_3\Dataset_3_final.csv'

# Cargamos los CSVs
file_1 = pd.read_csv(dataset_1)
file_2 = pd.read_csv(dataset_2)
file_3 = pd.read_csv(dataset_3)

#Calculamos las distribuvciones de las fotos en las cuatro clases
dataset_1_class = file_1['Class'].value_counts()
dataset_2_class = file_2['Class'].value_counts()
dataset_3_class = file_3['Class'].value_counts()

# # Gráfico de barras para el primer dataset
plt.figure(figsize=(10, 6))
sns.barplot(x=dataset_1_class.index, y=dataset_1_class.values)
plt.title('Dataset 1 - Image distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.show()

# Gráfico de barras para el segundo dataset
plt.figure(figsize=(10, 6))
sns.barplot(x=dataset_2_class.index, y=dataset_2_class.values)
plt.title('Dataset 2 - Image distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.show()

# Gráfico de barras para el segundo dataset
plt.figure(figsize=(10, 6))
sns.barplot(x=dataset_3_class.index, y=dataset_3_class.values)
plt.title('Dataset 3 - Image distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.show()

# Resultados numéricos
print('Resultados dataset 1')
print(dataset_1_class)
print('Total:' + str(file_1.shape[0]))

print('Resultados dataset 2')
print(dataset_2_class)
print('Total:' + str(file_2.shape[0]))

print('Resultados dataset 3')
print(dataset_3_class)
print('Total:' + str(file_3.shape[0]))