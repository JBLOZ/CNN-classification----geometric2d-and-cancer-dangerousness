%% skinCancerResNet50TransferLearning.m
% Script para realizar transfer learning sobre ResNet-50 y clasificar imágenes
% de cáncer de piel (maligno vs benigno).
%
% Estructura del dataset (skinCancerData.zip o carpetas ya extraídas):
%   - train: datos de entrenamiento
%   - val: datos de validación
%   - test: datos de prueba
%   - docs: (opcional) documentación o imágenes complementarias
%
% Cada carpeta "train", "val" y "test" debe contener dos subcarpetas:
%   malignant, benign
%
% Instrucciones:
%   1. Coloca "skinCancerData.zip" en el directorio actual O asegúrate de tener las carpetas
%      "train", "val" y "test" (y opcionalmente "docs") organizadas.
%   2. Si existe "trainedSkinCancerResNet50.mat", se carga el modelo; de lo contrario, se
%      carga la red preentrenada ResNet-50 (usando resnet50), se modifican
%      sus capas para clasificación binaria, se entrena y se guarda.
%   3. Se seleccionan 15 imágenes aleatorias del conjunto "test" y se muestra en la Command Window
%      la etiqueta predicha, la confianza y un OK (si es correcta) o FAIL (si no lo es).
%
% Requisitos:
%   - Deep Learning Toolbox
%   - Deep Learning Toolbox Model for ResNet-50 support package
%

clear; clc; close all;


%% Verificar disponibilidad de CUDA (GPU)
if gpuDeviceCount > 0
    try
        gpuDevice(1); % Selecciona la GPU 1
        executionEnv = 'gpu';
        fprintf('Dispositivo CUDA detectado. Se usará GPU para entrenamiento y pruebas.\n');
    catch ME
        executionEnv = 'cpu';
        fprintf('Error al intentar usar CUDA: %s\nSe usará CPU para entrenamiento y pruebas.\n', ME.message);
    end
else
    executionEnv = 'cpu';
    fprintf('No se detectó dispositivo CUDA. Se usará CPU para entrenamiento y pruebas.\n');
end
%% Definir rutas de carpetas
trainFolder = 'train';
valFolder   = 'val';
testFolder  = 'test';

%% COMPROBAR SI YA EXISTE EL MODELO ENTRENADO
if isfile('trainedSkinCancerResNet50.mat')
    load('trainedSkinCancerResNet50.mat', 'net');
    fprintf('Modelo de cáncer de piel cargado desde trainedSkinCancerResNet50.mat\n');
else
    %% Verificar si ya existen las carpetas necesarias
    if exist(trainFolder, 'dir') && exist(valFolder, 'dir') && exist(testFolder, 'dir')
        fprintf('Las carpetas train, val, test y docs ya existen. No se descomprime skinCancerData.zip.\n');
    else
        if exist('skinCancerData.zip','file')
            fprintf('Extrayendo skinCancerData.zip...\n');
            unzip('skinCancerData.zip'); % Extrae en el directorio actual
        else
            fprintf('No se encontró skinCancerData.zip. Se asume que las carpetas necesarias ya existen.\n');
        end
    end
    %% Visualizar algunas imágenes de entrenamiento
    figure;
    tiledlayout('flow');
    numImages = 20;
    perm = randperm(numel(imdsTrain.Files), numImages);
    for i = 1:numImages
        nexttile;
        img = imread(imdsTrain.Files{perm(i)});
        imshow(img);
        title(string(imdsTrain.Labels(perm(i))));
    end

    %% Crear datastores para entrenamiento y validación
    imdsTrain = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsVal   = imageDatastore(valFolder,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    %% Mostrar cantidad de imágenes por categoría en el conjunto de entrenamiento
    labelCount = countEachLabel(imdsTrain);
    disp('Cantidad de imágenes por categoría (entrenamiento):');
    disp(labelCount);
    
    %% Cargar la red preentrenada ResNet-50
    fprintf('Cargando ResNet-50 preentrenada...\n');
    baseNet = resnet50; % Carga la red preentrenada
    inputSize = baseNet.Layers(1).InputSize;  % Debe ser [224 224 3]
    
    % Convertir a layer graph para modificar la arquitectura
    lgraph = layerGraph(baseNet);
    
    % Definir las capas a eliminar: 'fc1000', 'fc1000_softmax' y la capa de clasificación final
    layersToRemove = {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'};
    lgraph = removeLayers(lgraph, layersToRemove);
    
    % Añadir nuevas capas para clasificación binaria (malignant vs benign)
    newLayers = [
        fullyConnectedLayer(2, 'Name', 'fc_skinCancer', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
        softmaxLayer('Name', 'softmax_skinCancer')
        classificationLayer('Name', 'ClassificationLayer_skinCancer')
    ];
    
    lgraph = addLayers(lgraph, newLayers);
    
    % Conectar las nuevas capas a la red
    % En ResNet-50, la última capa de pooling se llama 'avg_pool'
    lgraph = connectLayers(lgraph, 'avg_pool', 'fc_skinCancer');
    
    %% Crear augmentedImageDatastores para redimensionar las imágenes a [224 224]

    

%% Balancear el conjunto de entrenamiento
labelCountTrain = countEachLabel(imdsTrain);
minSetCountTrain = min(labelCountTrain.Count);
imdsTrainBalanced = splitEachLabel(imdsTrain, minSetCountTrain, 'randomize');

%% Balancear el conjunto de validación
labelCountVal = countEachLabel(imdsVal);
minSetCountVal = min(labelCountVal.Count);
imdsValBalanced = splitEachLabel(imdsVal, minSetCountVal, 'randomize');

%% Crear augmentedImageDatastores para redimensionar las imágenes a [224 224]
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrainBalanced);
augimdsVal   = augmentedImageDatastore(inputSize(1:2), imdsValBalanced);


    
    %% Opciones de entrenamiento
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', 4, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augimdsVal, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'training-progress');
    
    %% Entrenar la red
    fprintf('Entrenando la red para clasificación de cáncer de piel...\n');
    net = trainNetwork(augimdsTrain, lgraph, options);
    
    %% Guardar el modelo entrenado para usos futuros
    save('trainedSkinCancerResNet50.mat', 'net');
    fprintf('Modelo entrenado guardado en trainedSkinCancerResNet50.mat\n');
end

%% PROBAR EL MODELO EN 15 IMÁGENES ALEATORIAS DEL CONJUNTO DE TEST
fprintf('\nRealizando pruebas en 15 imágenes aleatorias del conjunto de test:\n');
inputSize = net.Layers(1).InputSize;
if ~isfolder(testFolder)
    error('La carpeta "test" no se encontró.');
end
imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numTestImages = 15;
perm = randperm(numel(imdsTest.Files), numTestImages);
for i = 1:numTestImages
    img = imread(imdsTest.Files{perm(i)});
    % Redimensionar la imagen al tamaño esperado (224x224)
    imgResized = imresize(img, inputSize(1:2));
    
    
    % Realizar la clasificación
    scores = predict(net, imgResized);
    maxScore = max(scores);
    

    if maxScore == scores(1)
        predictedLabel = 'benign';
    else
        predictedLabel = "malignant";
    end

    
    % Obtener la etiqueta real (según la estructura de carpetas)
    trueLabel = imdsTest.Labels(perm(i));
    
    
    % Comparar predicción y etiqueta real
    if predictedLabel == trueLabel
        status = 'OK';
    else
        status = 'FAIL';
    end
    
    % Imprimir el resultado
    fprintf('Imagen %d: Etiqueta predicha: %s, Confianza: %.2f%%, %s\n', ...
        i, predictedLabel, maxScore*100, status);
end

% Evaluar el modelo en todas las imágenes del conjunto de test
numTestImages = numel(imdsTest.Files);
benignTotal = 0;
benignCorrect = 0;
malignantTotal = 0;
malignantCorrect = 0;

for i = 1:numTestImages
    img = imread(imdsTest.Files{i});
    % Redimensionar la imagen al tamaño requerido
    imgResized = imresize(img, inputSize(1:2));
    
    % Realizar la clasificación
    scores = predict(net, imgResized);
    maxScore = max(scores);
    
    % Se asume que el índice 1 corresponde a 'benign'
    if maxScore == scores(1)
        predictedLabel = 'benign';
    else
        predictedLabel = 'malignant';
    end
    
    % Obtener la etiqueta verdadera
    trueLabel = imdsTest.Labels(i);
    
    % Acumular contadores según la etiqueta verdadera
    if strcmpi(char(trueLabel), 'benign')
        benignTotal = benignTotal + 1;
        if strcmpi(predictedLabel, 'benign')
            benignCorrect = benignCorrect + 1;
        end
    elseif strcmpi(char(trueLabel), 'malignant')
        malignantTotal = malignantTotal + 1;
        if strcmpi(predictedLabel, 'malignant')
            malignantCorrect = malignantCorrect + 1;
        end
    end
end

% Calcular porcentajes de acierto para cada clase
benignAccuracy = (benignCorrect / benignTotal) * 100;
malignantAccuracy = (malignantCorrect / malignantTotal) * 100;

% Mostrar resultados finales
fprintf('\nResultados finales en el conjunto de test:\n');
fprintf('Porcentaje de acierto en imágenes benignas: %.2f%%\n', benignAccuracy);
fprintf('Porcentaje de acierto en imágenes malignas: %.2f%%\n', malignantAccuracy);
