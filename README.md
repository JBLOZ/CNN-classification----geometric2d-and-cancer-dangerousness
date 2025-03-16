# Clasificación CNN: Formas Geométricas 2D y Clasificación de Cáncer de Piel

Este repositorio contiene una implementación completa de Redes Neuronales Convolucionales (CNNs) para tareas de clasificación de imágenes. El proyecto incluye dos componentes principales:

1. **Modelo Base Geométrico**: Una CNN construida desde cero para clasificar formas geométricas 2D
2. **Modelos de Clasificación de Cáncer de Piel**: Dos implementaciones avanzadas usando transfer learning con ResNet-50

## Tabla de Contenidos

- Estructura del Proyecto
- Requisitos
- Clasificación de Formas Geométricas
- Clasificación de Cáncer de Piel
  - Implementación Básica (V1)
  - Implementación Avanzada con Metadatos (V2)
- Preprocesamiento de Datos
- Evaluación de Rendimiento
- Aceleración GPU

## Estructura del Proyecto

```
CNN-classification----geometric2d-and-cancer-dangerousness/
├── geometric_base_model/
│   ├── train/                          # Conjunto de entrenamiento de formas geométricas
│   │   ├── circle/
│   │   ├── kite/
│   │   └── ...
│   ├── val/                            # Conjunto de validación de formas geométricas
│   ├── test/                           # Conjunto de prueba de formas geométricas
│   ├── trained2Dgeometricshapes.mat    # Modelo preentrenado
│   └── shapesDataClassifierIntegrated.m
│
├── resnet50_V(1)/
│   ├── train/                          # Conjunto de entrenamiento de imágenes de cáncer de piel
│   │   ├── benign/
│   │   └── malignant/
│   ├── val/                            # Conjunto de validación de imágenes de cáncer de piel
│   ├── test/                           # Conjunto de prueba de imágenes de cáncer de piel
│   ├── HAM10000_metadata.csv
│   ├── skinCancerResNet50.m
│   ├── trainedSkinCancerResNet50.mat   # Modelo preentrenado
│   └── division.py
│
└── resnet50_V(2)_CUDA_and_metadata/
    ├── train/                          # Conjunto de entrenamiento de imágenes de cáncer de piel
    │   ├── benign/
    │   └── malignant/
    ├── val/                            # Conjunto de validación de imágenes de cáncer de piel
    ├── test/                           # Conjunto de prueba de imágenes de cáncer de piel
    ├── HAM10000_metadata.csv
    ├── cudaSkinCancerResNet50_v2.m
    ├── trainedSkinCancerResNet50_v2.mat
    └── division.py
        
```

## Requisitos

- **MATLAB** (R2020b o más reciente recomendado)
  - Deep Learning Toolbox
  - Image Processing Toolbox
  - Paquete de soporte Deep Learning Toolbox Model for ResNet-50
- **Python** (para scripts de preprocesamiento de datos)
  - pandas
  - numpy
  - scikit-learn
  - shutil
- **GPU compatible con CUDA** (opcional, para entrenamiento acelerado)

## Clasificación de Formas Geométricas

El clasificador de formas geométricas es una CNN construida desde cero para identificar seis formas diferentes: círculo, cometa, paralelogramo, cuadrado, trapecio y triángulo.

### Arquitectura del Modelo

```matlab
layers = [
    imageInputLayer(imgSize)
    
    convolution2dLayer(3,8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3,16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3,32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
]
```

El modelo consta de tres bloques convolucionales con un número creciente de filtros (8 → 16 → 32), cada uno seguido de normalización por lotes, activación ReLU y max pooling. Las capas finales incluyen una capa totalmente conectada, activación softmax y salida de clasificación.

### Parámetros de Entrenamiento

```matlab
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsVal, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', executionEnv);
```

La red se entrena usando el optimizador Stochastic Gradient Descent with Momentum (SGDM) con una tasa de aprendizaje inicial de 0.01 durante 4 épocas.

### Uso

Para entrenar y evaluar el modelo de formas geométricas:

1. Coloca el conjunto de datos en las carpetas apropiadas o asegúrate de que `2DgeometricShapesData.zip` esté en el directorio de trabajo
2. Ejecuta el script shapesDataClassifierIntegrated.m
3. El script automáticamente:
   - Comprueba si existe un modelo pre-entrenado (`trained2Dgeometricshapes.mat`)
   - Si no lo encuentra, entrena un nuevo modelo y lo guarda
   - Evalúa el modelo en 15 imágenes aleatorias del conjunto de prueba

## Clasificación de Cáncer de Piel

El proyecto incluye dos implementaciones para la clasificación de cáncer de piel:

### Implementación Básica (V1)

La primera implementación utiliza transfer learning con ResNet-50 para clasificar lesiones cutáneas como benignas o malignas basándose únicamente en datos de imagen.

#### Modificación de ResNet-50

```matlab
% Modify the ResNet-50 architecture for binary classification
lgraph = layerGraph(baseNet);
layersToRemove = {'fc1000','fc1000_softmax'};
if any(strcmp({lgraph.Layers.Name},'ClassificationLayer_predictions'))
    layersToRemove{end+1} = 'ClassificationLayer_predictions';
elseif any(strcmp({lgraph.Layers.Name},'ClassificationLayer_fc1000'))
    layersToRemove{end+1} = 'ClassificationLayer_fc1000';
else
    error('No se encontró la capa de clasificación final en ResNet-50.');
end
lgraph = removeLayers(lgraph, layersToRemove);

% Add new layers for binary classification
newLayers = [
    fullyConnectedLayer(2, 'Name', 'fc_skinCancer', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
    softmaxLayer('Name', 'softmax_skinCancer')
    classificationLayer('Name', 'ClassificationLayer_skinCancer')
];
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'fc_skinCancer');
```

La ResNet-50 preentrenada se modifica eliminando las capas de clasificación finales y reemplazándolas con capas adecuadas para la clasificación binaria (benigno vs maligno).

#### Opciones de Entrenamiento

```matlab
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 4, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
```

El modelo se entrena usando SGDM con una tasa de aprendizaje pequeña (1e-4) para ajustar finamente la red preentrenada durante 4 épocas.

### Implementación Avanzada con Metadatos (V2)

La segunda implementación extiende la primera incorporando metadatos del paciente (edad y sexo) extraídos de los nombres de archivo de las imágenes para mejorar el rendimiento de la clasificación.

#### Arquitectura de Red con Integración de Metadatos

```matlab
% Convert to layer graph for architecture modification
lgraph = layerGraph(baseNet);
lgraph = removeLayers(lgraph, layersToRemove);

% Add flatten layer for the image branch
flattenLayerImage = flattenLayer('Name','flatten_img');
lgraph = addLayers(lgraph, flattenLayerImage);
lgraph = connectLayers(lgraph, 'avg_pool', 'flatten_img');

% Metadata branch
metaLayers = [
    featureInputLayer(2, 'Name','metadata_input','Normalization','none')
    fullyConnectedLayer(16, 'Name','fc_metadata','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu_metadata')
];
lgraph = addLayers(lgraph, metaLayers);

% Fusion and final classification branch
finalLayers = [
    concatenationLayer(1,2,'Name','concat')
    fullyConnectedLayer(2, 'Name','fc_skinCancer','WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax_skinCancer')
    classificationLayer('Name','ClassificationLayer_skinCancer')
];
lgraph = addLayers(lgraph, finalLayers);

% Connect both branches to the fusion layer
lgraph = connectLayers(lgraph, 'flatten_img', 'concat/in1');
lgraph = connectLayers(lgraph, 'relu_metadata', 'concat/in2');
```

Esta arquitectura tiene:
1. Una rama de procesamiento de imágenes (ResNet-50)
2. Una rama de procesamiento de metadatos (red totalmente conectada)
3. Una capa de fusión que combina características de ambas ramas
4. Capas finales de clasificación

#### Extracción de Metadatos

```matlab
% Function to extract metadata from the filename
function meta = parseMetadata(nameStr)
    % Split the name using the separator ''
    parts = split(nameStr, '');
    age_val = NaN;
    sex_val = NaN;
    for i = 1:numel(parts)
        part = parts{i};
        if startsWith(part, 'age-')
            ageStr = extractAfter(part, 'age-');
            age_val = str2double(ageStr);
        elseif startsWith(part, 'sex-')
            sexStr = lower(extractAfter(part, 'sex-'));
            if contains(sexStr, 'male')
                sex_val = 1;
            elseif contains(sexStr, 'female')
                sex_val = 0;
            else
                sex_val = 0.5;
            end
        end
    end
    if isnan(age_val)
        age_norm = 0;
    else
        age_norm = age_val / 100;
    end
    if isnan(sex_val)
        sex_val = 0.5;
    end
    % Return column vector (2×1)
    meta = [age_norm; sex_val];
end
```

La función de extracción de metadatos analiza el nombre de archivo de la imagen para extraer información de edad y sexo:
- La edad se normaliza dividiéndola por 100
- El sexo se codifica como 1 (masculino), 0 (femenino) o 0.5 (desconocido)

## Preprocesamiento de Datos

El proyecto incluye scripts de Python (`division.py`) para:

1. Dividir el conjunto de datos HAM10000 de cáncer de piel en conjuntos de entrenamiento, validación y prueba
2. Categorizar imágenes en clases benignas o malignas
3. Renombrar imágenes para incluir metadatos en el nombre del archivo

### Categorización de Imágenes

El script categoriza las lesiones cutáneas según el diagnóstico:
- **Malignas**: queratosis actínica (akiec), carcinoma basocelular (bcc), melanoma (mel)
- **Benignas**: queratosis benigna (bkl), nevos melanocíticos (nv), lesiones vasculares (vasc), dermatofibroma (df)

### Proporción de División del Conjunto de Datos

Los datos se dividen de la siguiente manera:
- **Conjunto de entrenamiento**: 70% de los datos
- **Conjunto de validación**: 15% de los datos
- **Conjunto de prueba**: 15% de los datos

### Extracción y Codificación de Metadatos

En la versión 2, los nombres de archivo de las imágenes incluyen metadatos codificados:
```
ISIC_0027419__dx-bkl__dx_type-histo__age-80.0__sex-male__localization-scalp__dataset-vidir_modern.jpg
```

Este formato permite que el modelo acceda a la información del paciente directamente desde el nombre del archivo sin necesidad de consultar el archivo CSV durante el entrenamiento o la inferencia.

## Aceleración GPU

Ambas implementaciones verifican la disponibilidad de GPUs compatibles con CUDA y las utilizan cuando están disponibles:

```matlab
if gpuDeviceCount > 0
    try
        gpuDevice(1); % Select GPU 1
        executionEnv = 'gpu';
        fprintf('CUDA device detected. GPU will be used for training and testing.\n');
    catch ME
        executionEnv = 'cpu';
        fprintf('Error trying to use CUDA: %s\nCPU will be used for training and testing.\n', ME.message);
    end
else
    executionEnv = 'cpu';
    fprintf('No CUDA device detected. CPU will be used for training and testing.\n');
end
```

El entorno de ejecución ('gpu' o 'cpu') se pasa a las opciones de entrenamiento para determinar si se utilizará la aceleración GPU.

## Evaluación de Rendimiento

Todos los modelos se evalúan probándolos en un subconjunto de los datos de prueba (15 imágenes aleatorias). La salida muestra:
- La etiqueta predicha
- La puntuación de confianza (probabilidad)
- Si la predicción fue correcta (OK) o incorrecta (FAIL)

Esto proporciona una evaluación rápida del rendimiento del modelo y permite una retroalimentación inmediata sobre la precisión de la clasificación.

---

Este proyecto demuestra un enfoque integral para la clasificación de imágenes utilizando CNNs, desde modelos básicos construidos desde cero hasta implementaciones avanzadas de transfer learning que incorporan metadatos adicionales. El código está diseñado para ser modular, bien documentado e incluye aceleración GPU para un mejor rendimiento.
