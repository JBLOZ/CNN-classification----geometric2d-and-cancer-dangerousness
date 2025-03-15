%% shapesDataClassifierIntegrated.m
% Script para entrenar (o cargar) una red neuronal que clasifica
% imágenes de formas geométricas y luego probar el modelo en 15 imágenes aleatorias.
%
% Estructura del dataset (2DgeometricShapesData.zip):
%   - train: datos de entrenamiento
%   - val: datos de validación
%   - test: datos de prueba
%
% Cada carpeta contiene subcarpetas con las categorías:
%   circle, kite, parallelogram, square, trapezoid, triangle
%
% Instrucciones:
%   1. Coloca "2DgeometricShapesData.zip" en el directorio actual de MATLAB o asegúrate de tener
%      las carpetas "train", "val" y "test" organizadas.
%   2. Si existe "trained2Dgeometricshapes.mat", se carga el modelo; de lo contrario, se
%      extrae 2DgeometricShapesData.zip (si las carpetas no existen), se entrena el modelo usando
%      los datos de "train" y "val", y se guarda en "trained2Dgeometricshapes.mat".
%   3. Finalmente, se seleccionan 15 imágenes aleatorias del conjunto "test" y se
%      muestra en la Command Window la etiqueta predicha, la confianza y se
%      indica OK si la predicción es correcta o FAIL en caso contrario.
%
% Requisitos: Deep Learning Toolbox.
%
% Autor: [Tu Nombre]
% Fecha: [Fecha]

clear; clc; close all;

%% Definir rutas de carpetas
trainFolder = 'train';
valFolder   = 'val';
testFolder  = 'test';

%% COMPROBAR SI YA EXISTE EL MODELO ENTRENADO
if isfile('trained2Dgeometricshapes.mat')
    load('trained2Dgeometricshapes.mat', 'net');
    fprintf('Modelo entrenado encontrado y cargado desde trained2Dgeometricshapes.mat\n');
else
    %% Verificar si las carpetas ya existen
    if exist(trainFolder, 'dir') && exist(valFolder, 'dir') && exist(testFolder, 'dir')
        fprintf('Las carpetas "train", "val" y "test" ya existen. No se descomprime 2DgeometricShapesData.zip.\n');
    else
        if exist('2DgeometricShapesData.zip', 'file')
            fprintf('Extrayendo 2DgeometricShapesData.zip...\n');
            unzip('2DgeometricShapesData.zip');  % Extrae el contenido del ZIP en el directorio actual
        else
            error('No se encontró 2DgeometricShapesData.zip. Colócalo en el directorio actual.');
        end
    end
    
    %% Crear datastores para entrenamiento, validación y prueba
    imdsTrain = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsVal   = imageDatastore(valFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsTest  = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    %% Visualizar algunas imágenes de entrenamiento
    figure;
    tiledlayout('flow');
    numImages = min(20, numel(imdsTrain.Files));
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
    
    %% Determinar el tamaño de la imagen de entrada
    img = readimage(imdsTrain,1);
    imgSize = size(img);
    if numel(imgSize) == 2
        % Si la imagen es en escala de grises, agregar dimensión del canal
        imgSize = [imgSize 1];
    end
    
    %% Definir la arquitectura de la red neuronal
    numClasses = numel(categories(imdsTrain.Labels));
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
        classificationLayer];
    
    %% Especificar las opciones de entrenamiento
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',4, ...
        'Shuffle','every-epoch', ...
        'ValidationData', imdsVal, ...
        'ValidationFrequency',30, ...
        'Plots','training-progress', ...
        'Verbose',false);
    
    %% Entrenar la red neuronal
    fprintf('Entrenando la red neuronal...\n');
    net = trainNetwork(imdsTrain, layers, options);
    
    %% Guardar el modelo entrenado para usos futuros
    save('trained2Dgeometricshapes.mat', 'net');
    fprintf('Modelo entrenado guardado en trained2Dgeometricshapes.mat\n');
end

%% Crear datastore para el conjunto de test (si no existe)
if ~isfolder(testFolder)
    error('La carpeta test no se encontró.');
end
imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Probar el modelo en 15 imágenes aleatorias del conjunto de test
fprintf('\nRealizando pruebas en 15 imágenes aleatorias del conjunto de test:\n');
inputSize = net.Layers(1).InputSize;
classNames = categories(imdsTest.Labels);

numTestImages = min(15, numel(imdsTest.Files));
perm = randperm(numel(imdsTest.Files), numTestImages);
for i = 1:numTestImages
    img = imread(imdsTest.Files{perm(i)});
    % Redimensionar la imagen al tamaño esperado por la red
    imgResized = imresize(img, inputSize(1:2));
    if size(imgResized,3) ~= inputSize(3)
        if inputSize(3)==1 && size(imgResized,3)==3
            imgResized = rgb2gray(imgResized);
        elseif inputSize(3)==3 && size(imgResized,3)==1
            imgResized = repmat(imgResized, [1 1 3]);
        end
    end
    
    % Realizar la predicción y obtener los scores
    scores = predict(net, imgResized);
    [maxScore, idx] = max(scores);
    predictedLabel = classNames(idx);
    
    % Obtener la etiqueta real del datastore
    trueLabel = imdsTest.Labels(perm(i));
    
    % Verificar si la predicción es correcta
    if predictedLabel == trueLabel
        status = 'OK';
    else
        status = 'FAIL';
    end
    
    % Imprimir la etiqueta predicha, la confianza y el estado
    fprintf('Imagen %d: Etiqueta predicha: %s, Confianza: %.2f%%, %s\n', ...
        i, string(predictedLabel), maxScore*100, status);
end
