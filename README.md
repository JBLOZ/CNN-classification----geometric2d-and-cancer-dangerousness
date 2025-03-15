# Proyectos de Clasificación de Imágenes en MATLAB

Este repositorio contiene dos proyectos de clasificación de imágenes desarrollados en MATLAB. Cada proyecto utiliza una red neuronal para abordar un problema de clasificación distinto:

- **Shapes Data Classifier**: Una red neuronal convolucional sencilla para clasificar imágenes de formas geométricas en 2D.
- **Skin Cancer Classification (Transfer Learning con ResNet-50)**: Un proyecto de transfer learning que adapta la red preentrenada ResNet-50 para clasificar imágenes de cáncer de piel en dos categorías: maligno y benigno.



## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Requisitos](#requisitos)
- [Instalación y Preparación del Dataset](#instalación-y-preparación-del-dataset)
- [Shapes Data Classifier](#shapes-data-classifier)
  - [Estructura del Dataset](#estructura-del-dataset-shapes)
  - [Arquitectura de la Red](#arquitectura-de-la-red-shapes)
  - [Uso del Programa](#uso-del-programa-shapes)
- [Skin Cancer Classification (Transfer Learning)](#skin-cancer-classification-transfer-learning)
  - [Dataset y Organización](#dataset-y-organización)
  - [Transfer Learning con ResNet-50](#transfer-learning-con-resnet-50)
  - [Uso del Programa](#uso-del-programa-skin-cancer)
- [Ejecución y Pruebas](#ejecución-y-pruebas)
- [Notas y Consideraciones Adicionales](#notas-y-consideraciones-adicionales)
- [Contacto](#contacto)

---

## Descripción General

Este repositorio ofrece ejemplos prácticos sobre cómo crear, entrenar y evaluar modelos de clasificación de imágenes en MATLAB. Se abordan dos problemas distintos:

1. **Clasificación de Formas Geométricas (2D):**  
   Se utiliza una red neuronal convolucional construida desde cero para identificar y clasificar imágenes de formas geométricas, tales como círculos, cometas, paralelogramos, cuadrados, trapecios y triángulos. Este proyecto es ideal para comprender los fundamentos del diseño y entrenamiento de CNNs simples.

2. **Clasificación de Cáncer de Piel (Transfer Learning):**  
   Se aplica transfer learning usando la red preentrenada ResNet-50, originalmente entrenada en ImageNet, para distinguir entre lesiones malignas y benignas en imágenes dermatoscópicas. Se utiliza el dataset HAM10000 (u otro similar organizado en carpetas) para entrenar y evaluar el modelo.

---

## Requisitos

- **MATLAB** con el [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html).
- Para el proyecto de *Transfer Learning*:
  - [Deep Learning Toolbox Model for ResNet-50 support package](https://www.mathworks.com/matlabcentral/fileexchange/42950-resnet-50-pretrained-network)
- Conexión a internet para descargar datasets y paquetes adicionales (si es necesario).

---

## Instalación y Preparación del Dataset

### Shapes Data Classifier
- **Dataset:**  
  El dataset se encuentra en el archivo `2DgeometricShapesData.zip` (16.2 MB).  
- **Estructura Esperada:**  
  Al ejecutar el script, si las carpetas `train`, `val` y `test` no existen, el programa extraerá `2DgeometricShapesData.zip`.  
- **Organización Ideal:**
### Skin Cancer Classification
- **Dataset:**  
Se espera que tengas organizadas las carpetas `train`, `val` y `test` para el problema de cáncer de piel.  
- **Organización:**  
Cada una de estas carpetas debe contener dos subcarpetas:
- `malignant`
- `benign`

---

## Shapes Data Classifier

### Estructura del Dataset (Shapes)
El dataset para formas geométricas debe estar organizado en carpetas separadas para cada clase. El script `shapesDataClassifierIntegrated.m` se encarga de:
- Verificar la existencia del archivo `2DgeometricShapesData.zip` y de las carpetas `train`, `val` y `test`.
- Extraer el dataset si las carpetas no existen.
- Crear datastores de imágenes a partir de estas carpetas.

### Arquitectura de la Red
La red neuronal para clasificar las formas se construye de forma sencilla:
- **Capa de Entrada:**  
Ajusta el tamaño de la imagen según la primera imagen del conjunto, soportando imágenes en escala de grises o en color.
- **Capas Convolucionales y de Pooling:**  
La red consta de tres bloques, cada uno compuesto por una capa convolucional, normalización por lotes y una capa ReLU, seguidos por capas de max pooling para reducir la dimensión espacial.
- **Capas Finales:**  
Una capa totalmente conectada con un número de salidas igual al número de clases, seguida de una capa softmax y una capa de clasificación.

### Uso del Programa (Shapes)
- Ejecuta el script `shapesDataClassifierIntegrated.m` en MATLAB.
- Si el archivo `trained2Dgeometricshapes.mat` ya existe, se cargará el modelo; de lo contrario, se extraerá `2DgeometricShapesData.zip` (si es necesario) y se entrenará la red.
- Tras el entrenamiento, el script selecciona 15 imágenes aleatorias del conjunto `test` y muestra en la Command Window la etiqueta predicha, la confianza y el resultado (OK/FAIL).

---

## Skin Cancer Classification (Transfer Learning)

### Dataset y Organización
El dataset para la clasificación de cáncer de piel (por ejemplo, HAM10000 o un dataset similar) debe estar organizado en carpetas `train`, `val` y `test`, cada una con dos subcarpetas: `malignant` y `benign`.

### Transfer Learning con ResNet-50
Para abordar el problema de clasificación de cáncer de piel, se utiliza transfer learning:
- Se carga la red preentrenada **ResNet-50** utilizando la función `resnet50`.
- Se transforma la red en un `layerGraph` para eliminar las últimas capas (originalmente diseñadas para 1000 clases de ImageNet).
- Se añaden nuevas capas (una capa totalmente conectada, softmax y una capa de clasificación) para obtener una salida binaria.
- Se utilizan `augmentedImageDatastore` para redimensionar automáticamente las imágenes al tamaño requerido (224×224×3) por ResNet-50.
- Se entrena la red utilizando las imágenes de entrenamiento y se valida con el conjunto de validación.
- El modelo entrenado se guarda en `trainedSkinCancerResNet50.mat` para evitar reentrenamientos innecesarios.

### Uso del Programa (Skin Cancer)
- Ejecuta el script `skinCancerResNet50TransferLearning.m` en MATLAB.
- Si el archivo `trainedSkinCancerResNet50.mat` ya existe, se carga el modelo; de lo contrario, se extrae `skinCancerData.zip` (si es necesario) y se entrena el modelo.
- El script crea datastores y redimensiona las imágenes.
- Finalmente, se realizan 15 pruebas aleatorias en el conjunto `test`, mostrando la etiqueta predicha, la confianza y si la predicción es correcta (OK/FAIL).

---

## Ejecución y Pruebas

Para ejecutar cada proyecto, sigue estos pasos:

1. **Clona el repositorio** en tu máquina.
2. **Coloca los datasets correspondientes**:
 - Para **Shapes Data Classifier**: Coloca `2DgeometricShapesData.zip` en la raíz o asegúrate de que las carpetas `train`, `val` y `test` estén correctamente organizadas.
 - Para **Skin Cancer Classification**: Organiza las carpetas `train`, `val` y `test` con las subcarpetas `malignant` y `benign`, o coloca `skinCancerData.zip` en la raíz.
3. Abre MATLAB y navega hasta el directorio del proyecto.
4. Ejecuta el script correspondiente:
 - Para **Shapes Data Classifier**:  
   ```matlab
   shapesDataClassifierIntegrated
   ```
   El modelo entrenado se guardará como `trained2Dgeometricshapes.mat`.
 - Para **Skin Cancer Classification**:  
   ```matlab
   skinCancerResNet50TransferLearning
   ```
   El modelo entrenado se guardará como `trainedSkinCancerResNet50.mat`.
5. Revisa la Command Window para ver los resultados del entrenamiento y de las pruebas aleatorias.

---

## Notas y Consideraciones Adicionales

- **Modelos Preentrenados y Transfer Learning:**  
El uso de redes preentrenadas como ResNet-50 permite aprovechar características ya aprendidas de grandes conjuntos de datos (ImageNet) y adaptarlas a problemas específicos, reduciendo el tiempo de entrenamiento y mejorando la precisión.

- **Optimización de Imágenes:**  
Se utilizan `augmentedImageDatastore` para redimensionar imágenes en tiempo real, lo que facilita el manejo de datasets con imágenes de tamaños variados y garantiza la compatibilidad con la arquitectura de la red.

- **Almacenamiento del Modelo:**  
Una vez entrenado, el modelo se guarda en un archivo `.mat` (ya sea `trained2Dgeometricshapes.mat` o `trainedSkinCancerResNet50.mat`), evitando reentrenamientos innecesarios y permitiendo su reutilización en futuras ejecuciones.

- **Evaluación del Modelo:**  
Cada script incluye un bloque de pruebas que selecciona 15 imágenes aleatorias del conjunto de test y muestra los resultados de la predicción (etiqueta, confianza y verificación de exactitud).


Este repositorio es una muestra práctica del uso de redes neuronales en MATLAB para la clasificación de imágenes, abarcando tanto soluciones simples (clasificación de formas) como técnicas avanzadas de transfer learning (clasificación de cáncer de piel). ¡Espero que te resulte de gran utilidad!

