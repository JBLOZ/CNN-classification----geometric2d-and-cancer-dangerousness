#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para dividir el dataset de imágenes en train, validation y test,
organizando además en subcarpetas según sean cánceres malignos o benignos.
Se renombra cada imagen incluyendo en el nombre algunos metadatos (dx_type, age, sex, localization y dataset)
para poder usarlos posteriormente en la CNN sin necesidad de usar el CSV en cada paso.
"""

import os
import shutil
import pandas as pd
import numpy as np

# ----- CONFIGURACIÓN -----
# Ruta al CSV de metadatos
CSV_PATH = "HAM10000_metadata.csv"

# Carpeta donde se encuentran todas las imágenes (se espera que el nombre de la imagen sea image_id + ".jpg")
IMAGES_SOURCE_DIR = "alldata"

# Carpeta de salida donde se crearán las subcarpetas train, val y test
OUTPUT_DIR = "."

# Porcentajes para cada subconjunto
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# El resto será TEST

# Definir etiquetas maligno vs benigno.
# En este ejemplo se consideran malignos: akiec, bcc, mel; benignos el resto.
MALIGNANT_LABELS = {"akiec", "bcc", "mel"}

# Semilla para reproducibilidad
RANDOM_SEED = 42

# ----- FIN CONFIGURACIÓN -----

def create_folder_structure(base_dir):
    """Crea la estructura de carpetas:
       base_dir/train/malignant, base_dir/train/benign,
       base_dir/val/malignant, base_dir/val/benign,
       base_dir/test/malignant, base_dir/test/benign
    """
    for split in ["train", "val", "test"]:
        for label in ["malignant", "benign"]:
            dir_path = os.path.join(base_dir, split, label)
            os.makedirs(dir_path, exist_ok=True)
    print("Estructura de carpetas creada en:", base_dir)

def get_class_label(dx):
    """Devuelve 'malignant' si dx está en MALIGNANT_LABELS, y 'benign' en otro caso."""
    return "malignant" if dx.lower() in MALIGNANT_LABELS else "benign"

def rename_with_metadata(row):
    """
    Construye un nuevo nombre de fichero que incorpora image_id y metadatos.
    Se concatenan el image_id, dx, dx_type, age, sex, localization y dataset.
    Ejemplo de resultado:
    ISIC_0027419__dx-bkl__dx_type-histo__age-80.0__sex-male__localization-scalp__dataset-vidir_modern.jpg
    """
    image_id = row["image_id"]
    dx = row["dx"]
    dx_type = row["dx_type"]
    age = row["age"]
    sex = row["sex"]
    localization = row["localization"]
    dataset_info = row["dataset"]
    
    # Reemplazamos espacios u otros caracteres problemáticos, si es necesario
    new_filename = f"{image_id}__dx-{dx}__dx_type-{dx_type}__age-{age}__sex-{sex}__localization-{localization}__dataset-{dataset_info}.jpg"
    return new_filename

def split_dataset(df):
    """Divide el DataFrame en train, val y test según los porcentajes definidos."""
    df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    n_total = len(df_shuffled)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    
    df_train = df_shuffled.iloc[:n_train].copy()
    df_val   = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test  = df_shuffled.iloc[n_train+n_val:].copy()
    
    print(f"Total de muestras: {n_total} - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    return {"train": df_train, "val": df_val, "test": df_test}

def main():
    # Lee el CSV
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print("Error al leer el CSV:", e)
        return

    # Verifica que existan los campos necesarios
    required_columns = {"image_id", "dx", "dx_type", "age", "sex", "localization", "dataset"}
    if not required_columns.issubset(df.columns):
        print("Error: El CSV no contiene todas las columnas requeridas:", required_columns)
        return

    # Divide el dataset
    splits = split_dataset(df)
    
    # Crea la estructura de carpetas de salida
    create_folder_structure(OUTPUT_DIR)
    
    # Procesa cada partición
    for split_name, split_df in splits.items():
        print(f"\nProcesando split: {split_name}")
        for idx, row in split_df.iterrows():
            image_id = row["image_id"]
            # Se asume que las imágenes tienen extensión .jpg; modificar si es necesario.
            source_file = os.path.join(IMAGES_SOURCE_DIR, f"{image_id}.jpg")
            if not os.path.exists(source_file):
                print(f"Advertencia: La imagen {source_file} no existe. Saltándola.")
                continue

            # Determina la clase (malignant o benign)
            class_label = get_class_label(row["dx"])
            
            # Construye el nuevo nombre de fichero que incluye los metadatos
            new_filename = rename_with_metadata(row)
            
            # Define la ruta destino: OUTPUT_DIR/split/class_label/new_filename
            dest_dir = os.path.join(OUTPUT_DIR, split_name, class_label)
            dest_file = os.path.join(dest_dir, new_filename)
            
            try:
                shutil.copy2(source_file, dest_file)
            except Exception as e:
                print(f"Error copiando {source_file} a {dest_file}: {e}")

    print("\nDivisión y copia de imágenes completada.")

if __name__ == "__main__":
    main()