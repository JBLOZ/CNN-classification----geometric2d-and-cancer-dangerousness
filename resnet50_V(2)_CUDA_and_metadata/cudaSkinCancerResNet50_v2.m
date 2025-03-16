% skinCancerResNet50TransferLearning.m
%
% Script para realizar transfer learning sobre ResNet‑50 y clasificar imágenes
% de cáncer de piel (maligno vs benigno) incorporando metadatos extraídos
% del nombre de cada imagen (por ejemplo, “age” y “sex”).
%
% Se espera que el dataset esté organizado en carpetas:
% - train: datos de entrenamiento
% - val: datos de validación
% - test: datos de prueba
%
% Cada carpeta contiene dos subcarpetas: malignant y benign.
%
% NOTA: Las imágenes se han renombrado con el siguiente formato (ejemplo):
% ISIC_0024307__dx-nv__dx_type-follow_up__age-50.0__sex-male__localization-lower extremity__dataset-vidir_molemax.jpg
%
% De este nombre se extraen los metadatos “age” y “sex”. La edad se normaliza
% (dividiéndola por 100) y el sexo se codifica (male -> 1, female -> 0).
%
% Si ya existe "trainedSkinCancerResNet50_v2.mat" se carga el modelo; de lo contrario,
% se entrena la red (con incorporación de la rama de metadatos), se guarda y se
% realizan pruebas sobre 15 imágenes del conjunto de test.
%
% Requisitos:
% - Deep Learning Toolbox
% - Deep Learning Toolbox Model for ResNet-50 support package
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% Parámetros y tamaño objetivo de imagen (para ResNet‑50 es [224 224 3])
TARGET_SIZE = [224 224];

%% Crear datastores para entrenamiento, validación y prueba
trainFolder = 'train';
valFolder = 'val';
testFolder = 'test';
imdsTrain = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTrain.ReadFcn = @(f) readFcnMetadata(f, TARGET_SIZE); % Devuelve {imagen, metadatos}
imdsVal = imageDatastore(valFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsVal.ReadFcn = @(f) readFcnMetadata(f, TARGET_SIZE);
imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest.ReadFcn = @(f) readFcnMetadata(f, TARGET_SIZE);

%% Preparar un datastore combinado para entrenamiento y validación
dsTrain = combine(imdsTrain, arrayDatastore(imdsTrain.Labels));
dsTrain = transform(dsTrain, @mergeData);
dsVal = combine(imdsVal, arrayDatastore(imdsVal.Labels));
dsVal = transform(dsVal, @mergeData);

%% COMPROBAR SI YA EXISTE EL MODELO ENTRENADO CON METADATOS
if isfile('trainedSkinCancerResNet50_v2.mat')
    load('trainedSkinCancerResNet50_v2.mat', 'net');
    fprintf('Modelo de cáncer de piel (con metadatos) cargado desde trainedSkinCancerResNet50_v2.mat\n');
    inputSize = net.Layers(1).InputSize; % Para la rama de imagen
else
    %% Visualizar algunas imágenes de entrenamiento (sólo imágenes)
    figure;
    tiledlayout('flow');
    numImages = min(20, numel(imdsTrain.Files));
    perm = randperm(numel(imdsTrain.Files), numImages);
    for i = 1:numImages
        nexttile;
        data = readFcnMetadata(imdsTrain.Files{perm(i)}, TARGET_SIZE);
        img = data{1};
        imshow(img);
        title(string(imdsTrain.Labels(perm(i))));
    end
    %% Mostrar cantidad de imágenes por categoría en el conjunto de entrenamiento
    labelCount = countEachLabel(imdsTrain);
    disp('Cantidad de imágenes por categoría (entrenamiento):');
    disp(labelCount);

    %% Preparar la red preentrenada ResNet‑50 y modificar la arquitectura
    fprintf('Cargando ResNet-50 preentrenada...\n');
    baseNet = resnet50; % Carga la red preentrenada
    inputSize = baseNet.Layers(1).InputSize; % [224 224 3]
    % Convertir a layer graph para modificar la arquitectura
    lgraph = layerGraph(baseNet);
    % Eliminar las capas originales de clasificación
    layersToRemove = {'fc1000','fc1000_softmax'};
    if any(strcmp({lgraph.Layers.Name},'ClassificationLayer_predictions'))
        layersToRemove{end+1} = 'ClassificationLayer_predictions';
    elseif any(strcmp({lgraph.Layers.Name},'ClassificationLayer_fc1000'))
        layersToRemove{end+1} = 'ClassificationLayer_fc1000';
    else
        error('No se encontró la capa de clasificación final en ResNet-50.');
    end
    lgraph = removeLayers(lgraph, layersToRemove);
    % Añadir capa de flatten para la rama de imagen
    flattenLayerImage = flattenLayer('Name','flatten_img');
    lgraph = addLayers(lgraph, flattenLayerImage);
    lgraph = connectLayers(lgraph, 'avg_pool', 'flatten_img');
    % --- Rama para los metadatos ---
    metaLayers = [
        featureInputLayer(2, 'Name','metadata_input','Normalization','none')
        fullyConnectedLayer(16, 'Name','fc_metadata','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        reluLayer('Name','relu_metadata')
    ];
    lgraph = addLayers(lgraph, metaLayers);
    % --- Rama de fusión y clasificación final ---
    finalLayers = [
        concatenationLayer(1,2,'Name','concat')
        fullyConnectedLayer(2, 'Name','fc_skinCancer','WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax_skinCancer')
        classificationLayer('Name','ClassificationLayer_skinCancer')
    ];
    lgraph = addLayers(lgraph, finalLayers);
    % Conectar ambas ramas a la fusión:
    lgraph = connectLayers(lgraph, 'flatten_img', 'concat/in1');
    lgraph = connectLayers(lgraph, 'relu_metadata', 'concat/in2');

    %% Opciones de entrenamiento
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', 4, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', dsVal, ... % Datastore combinado para validación
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', executionEnv);

    %% Entrenar la red
    fprintf('Entrenando la red para clasificación de cáncer de piel (con metadatos)...\n');
    net = trainNetwork(dsTrain, lgraph, options);

    %% Guardar el modelo entrenado
    save('trainedSkinCancerResNet50_v2.mat', 'net');
    fprintf('Modelo entrenado (con metadatos) guardado en trainedSkinCancerResNet50_v2.mat\n');
end

%% PROBAR EL MODELO EN 15 IMÁGENES ALEATORIAS DEL CONJUNTO DE TEST
fprintf('\nRealizando pruebas en 15 imágenes aleatorias del conjunto de test:\n');
numTestImages = min(15, numel(imdsTest.Files));
perm = randperm(numel(imdsTest.Files), numTestImages);

% Crear un datastore combinado para pruebas, similar al usado en entrenamiento
dsTestSelected = subset(imdsTest, perm);
dsTestLabels = arrayDatastore(imdsTest.Labels(perm));
dsTest = combine(dsTestSelected, dsTestLabels);
dsTest = transform(dsTest, @mergeData);

% Realizar predicciones utilizando el datastore transformado
yPred = classify(net, dsTest, 'ExecutionEnvironment', executionEnv);
scores = predict(net, dsTest, 'ExecutionEnvironment', executionEnv);

% Obtener etiquetas reales
trueLabels = imdsTest.Labels(perm);

% Mostrar los resultados
for i = 1:numTestImages
    trueLabel = trueLabels(i);
    predictedLabel = yPred(i);
    score = scores(i,:);
    maxScore = max(score);
    
    % Comparar predicción y etiqueta real
    if predictedLabel == trueLabel
        status = 'OK';
    else
        status = 'FAIL';
    end
    
    % Imprimir el resultado
    fprintf('Imagen %d: Etiqueta predicha: %s, Confianza: %.2f%%, %s\n', ...
        i, char(predictedLabel), maxScore*100, status);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Función personalizada para leer imagen y extraer metadatos
function data = readFcnMetadata(filename, targetSize)
    % Leer la imagen
    img = imread(filename);
    % Si la imagen es en escala de grises, convertir a RGB
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    % Redimensionar la imagen al tamaño targetSize (altura x ancho)
    img = imresize(img, targetSize);
    % Extraer el nombre del archivo (sin ruta ni extensión)
    [~, name, ~] = fileparts(filename);
    % Extraer metadatos (edad y sexo) a partir del nombre del archivo
    metadata = parseMetadata(name);
    % Regresar un cell con la imagen y el vector de metadatos
    data = {img, metadata};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Función para extraer metadatos del nombre del archivo
function meta = parseMetadata(nameStr)
    % Dividir el nombre usando el separador ''
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
    % Retornar vector columna (2×1)
    meta = [age_norm; sex_val];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Función de transformación para combinar la salida de los datastores
function mergedData = mergeData(data)
    % data{1} es el output del imageDatastore: {imagen, metadata}
    % data{2} es la etiqueta (del arrayDatastore)
    img_meta = data{1};
    label = data{2};
    mergedData = {img_meta{1}, img_meta{2}, label};
end