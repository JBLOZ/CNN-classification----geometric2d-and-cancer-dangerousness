function skinCancerResNet50TransferLearning(mode)
% skinCancerResNet50TransferLearning(mode)
%   Ejecuta el script en modo 'train' o 'test'.
%
% Modo 'train': se verifica la existencia de las carpetas del dataset.
% Si no existen, se busca el archivo zip; y si no está, se descarga desde GitHub.
% Se entrena la red (si no existe el modelo entrenado) y se guarda.
%
% Modo 'test': si no existe el modelo entrenado (.mat) se descarga desde GitHub.
% Se carga el modelo y se ejecutan las pruebas sobre el conjunto de test.


if nargin < 1
    mode = 'train'; % Valor por defecto
end
mode = lower(mode);

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

switch mode
    case 'test'
        % Si estamos en modo test, comprobamos si existe el modelo entrenado
        if ~isfile('trainedSkinCancerResNet50.mat')
            fprintf('No se encontró trainedSkinCancerResNet50.mat.\n');
            fprintf('Descargando modelo desde GitHub release...\n');
            % Coloca la URL de tu release en GitHub para el modelo entrenado:
            modelURL = 'https://github.com/tu_usuario/tu_repo/releases/download/v1.0/trainedSkinCancerResNet50.mat';
            websave('trainedSkinCancerResNet50.mat', modelURL);
        end
        load('trainedSkinCancerResNet50.mat', 'net');
        fprintf('Modelo cargado desde trainedSkinCancerResNet50.mat\n');
        
    case 'train'
        % Primero se comprueba la existencia de las carpetas necesarias
        if exist(trainFolder, 'dir') && exist(valFolder, 'dir') && exist(testFolder, 'dir')
            fprintf('Las carpetas train, val y test ya existen.\n');
        else
            if exist('skinCancerData.zip','file')
                fprintf('Extrayendo skinCancerData.zip...\n');
                unzip('skinCancerData.zip'); % Extrae en el directorio actual
            else
                fprintf('No se encontró skinCancerData.zip.\n');
                fprintf('Descargando skinCancerData.zip desde GitHub release...\n');
                % Coloca la URL de tu release en GitHub para el dataset:
                dataURL = 'https://github.com/tu_usuario/tu_repo/releases/download/v1.0/skinCancerData.zip';
                websave('skinCancerData.zip', dataURL);
                unzip('skinCancerData.zip');
            end
        end
        
        % Si ya existe el modelo entrenado, lo cargamos; de lo contrario se entrena.
        if isfile('trainedSkinCancerResNet50.mat')
            fprintf('Modelo entrenado ya existe. Se carga el modelo.\n');
            load('trainedSkinCancerResNet50.mat', 'net');
        else
            %% Crear datastores para entrenamiento y validación
            imdsTrain = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            imdsVal   = imageDatastore(valFolder,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            
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
            
            %% Mostrar cantidad de imágenes por categoría en el conjunto de entrenamiento
            labelCount = countEachLabel(imdsTrain);
            disp('Cantidad de imágenes por categoría (entrenamiento):');
            disp(labelCount);
            
            %% Cargar la red preentrenada ResNet-50
            fprintf('Cargando ResNet-50 preentrenada...\n');
            baseNet = resnet50; % Carga la red preentrenada
            inputSize = baseNet.Layers(1).InputSize;  % [224 224 3]
            
            % Convertir a layer graph para modificar la arquitectura
            lgraph = layerGraph(baseNet);
            
            % Eliminar las capas finales de la red original
            layersToRemove = {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'};
            lgraph = removeLayers(lgraph, layersToRemove);
            
            % Añadir nuevas capas para clasificación binaria
            newLayers = [
                fullyConnectedLayer(2, 'Name', 'fc_skinCancer', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
                softmaxLayer('Name', 'softmax_skinCancer')
                classificationLayer('Name', 'ClassificationLayer_skinCancer')
            ];
            lgraph = addLayers(lgraph, newLayers);
            
            % Conectar las nuevas capas: la última capa de pooling en ResNet-50 es 'avg_pool'
            lgraph = connectLayers(lgraph, 'avg_pool', 'fc_skinCancer');
            
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
        
    otherwise
        error('Modo desconocido. Usa ''train'' o ''test''.');
end

%% PROBAR EL MODELO EN 15 IMÁGENES ALEATORIAS DEL CONJUNTO DE TEST
if ~isfolder(testFolder)
    error('La carpeta "test" no se encontró.');
end

fprintf('\nRealizando pruebas en 15 imágenes aleatorias del conjunto de test:\n');
imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
inputSize = net.Layers(1).InputSize;
numTestImages = 15;
perm = randperm(numel(imdsTest.Files), numTestImages);
for i = 1:numTestImages
    img = imread(imdsTest.Files{perm(i)});
    % Redimensionar la imagen al tamaño esperado (224x224)
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
    
    % Obtener la etiqueta real (según la estructura de carpetas)
    trueLabel = imdsTest.Labels(perm(i));
    
    % Comparar predicción y etiqueta real
    if strcmpi(char(trueLabel), predictedLabel)
        status = 'OK';
    else
        status = 'FAIL';
    end
    
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
    imgResized = imresize(img, inputSize(1:2));
    
    scores = predict(net, imgResized);
    maxScore = max(scores);
    
    if maxScore == scores(1)
        predictedLabel = 'benign';
    else
        predictedLabel = 'malignant';
    end
    
    trueLabel = imdsTest.Labels(i);
    
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

benignAccuracy = (benignCorrect / benignTotal) * 100;
malignantAccuracy = (malignantCorrect / malignantTotal) * 100;

fprintf('\nResultados finales en el conjunto de test:\n');
fprintf('Porcentaje de acierto en imágenes benignas: %.2f%%\n', benignAccuracy);
fprintf('Porcentaje de acierto en imágenes malignas: %.2f%%\n', malignantAccuracy);

end
