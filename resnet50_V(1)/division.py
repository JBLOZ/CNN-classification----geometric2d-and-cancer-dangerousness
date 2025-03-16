import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# --- Configuración ---
metadata_file = 'HAM10000_metadata.csv'  # Archivo de metadata
images_folder = 'alldata'         # Carpeta que contiene todas las imágenes

# Definir las clases según diagnóstico
malignant_classes = ['akiec', 'bcc', 'mel']
benign_classes = ['bkl', 'nv', 'vasc', 'df']

# --- Leer y procesar la metadata ---
df = pd.read_csv(metadata_file)

# Función para asignar clase según diagnóstico
def map_diagnosis(dx):
    if dx in malignant_classes:
        return 'malignant'
    elif dx in benign_classes:
        return 'benign'
    else:
        return 'unknown'

# Crear una nueva columna "class"
df['class'] = df['dx'].apply(map_diagnosis)
# Filtrar imágenes con diagnósticos desconocidos (si hubiera)
df = df[df['class'] != 'unknown']

# Suponer que los nombres de imagen son: <image_id>.jpg
df['filename'] = df['image_id'].astype(str) + '.jpg'

# --- División de los datos ---
# División estratificada para mantener la proporción de cada clase.
# Primero, se separa 70% para train y 30% para un conjunto temporal.
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['class'])
# Luego, se divide el conjunto temporal en val y test (50% cada uno => 15% de val, 15% de test)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['class'])

print("Número de imágenes - Train:", len(train_df))
print("Número de imágenes - Validation:", len(val_df))
print("Número de imágenes - Test:", len(test_df))

# --- Crear la estructura de carpetas ---
# Se crearán las carpetas: train/malignant, train/benign, val/malignant, val/benign, test/malignant, test/benign
splits = {'train': train_df, 'val': val_df, 'test': test_df}
for split_name in splits.keys():
    for cls in ['malignant', 'benign']:
        dir_path = os.path.join(split_name, cls)
        os.makedirs(dir_path, exist_ok=True)

# --- Copiar imágenes a las carpetas correspondientes ---
for split_name, split_df in splits.items():
    for idx, row in split_df.iterrows():
        src = os.path.join(images_folder, row['filename'])
        dest = os.path.join(split_name, row['class'], row['filename'])
        if os.path.exists(src):
            shutil.copy(src, dest)
        else:
            print(f"Archivo {src} no encontrado.")

print("¡División y organización completada!")