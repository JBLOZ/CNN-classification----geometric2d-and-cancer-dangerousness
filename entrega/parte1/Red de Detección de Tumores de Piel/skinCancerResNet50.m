function skinCancerResNet50TransferLearning(mode)
    % skinCancerResNet50TransferLearning(mode)
    %   Ejecuta el script en modo 'train' o 'test'.
    %
    % Modo 'train':
    %   - Verifica si existen las carpetas (train, val, test). Si no, busca skinCancerData.zip;
    %     y si tampoco está, lo descarga desde GitHub.
    %   - Si no existe el modelo entrenado, lo entrena y lo guarda.
    %
    % Modo 'test':
    %   - Si no existe el modelo entrenado (.mat), lo descarga desde GitHub.
    %   - Ejecuta solo las pruebas sobre el conjunto de test.
    %
    % RECUERDA: Sustituir las URLs de ejemplo con la de tus releases en GitHub.
    
    if nargin < 1
        mode = 'test'; % Valor por defecto
    end
    
    clc; close all;
    
    %% Verificar disponibilidad de CUDA (GPU)
    if gpuDeviceCount > 0
        try
            gpuDevice(1); % Selecciona la GPU 1
            executionEnv = 'gpu';
            fprintf('Dispositivo CUDA detectado. Se usará GPU.\n');
        catch ME
            executionEnv = 'cpu';
            fprintf('Error al intentar usar CUDA: %s\nSe usará CPU.\n', ME.message);
        end
    else
        executionEnv = 'cpu';
        fprintf('No se detectó GPU. Se usará CPU.\n');
    end
    
    %% Definir rutas de carpetas
    trainFolder = 'train';
    valFolder   = 'val';
    testFolder  = 'test';
    
    switch mode
        case 'train'
            %% Verificar existencia de carpetas
            if exist(trainFolder, 'dir') && exist(valFolder, 'dir') && exist(testFolder, 'dir')
                fprintf('Las carpetas train, val y test ya existen.\n');
            else
                if exist('skinCancerData.zip','file')
                    fprintf('Extrayendo skinCancerData.zip...\n');
                    unzip('skinCancerData.zip'); % Extrae en el directorio actual
                else
                    fprintf('No se encontró skinCancerData.zip.\n');
                    fprintf('Descargando skinCancerData.zip desde GitHub release...\n');
                    
                    dataURL = 'https://github.com/JBLOZ/CNN-classification----geometric2d-and-cancer-dangerousness/releases/download/v(3.0.0)/skinCancerData.zip';
                    websave('skinCancerData.zip', dataURL);

                    unzip('skinCancerData.zip');
                end
            end

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
            
            %% Mostrar cantidad de imágenes por categoría (entrenamiento)
            labelCount = countEachLabel(imdsTrain);
            disp('Cantidad de imágenes por categoría (entrenamiento):');
            disp(labelCount);
            
            %% Cargar la red preentrenada ResNet-50
            fprintf('Cargando ResNet-50 preentrenada...\n');
            baseNet = resnet50; % Carga la red preentrenada
            inputSize = baseNet.Layers(1).InputSize;  % [224 224 3]
            
            % Convertir a layer graph para modificar la arquitectura
            lgraph = layerGraph(baseNet);
            
            % Eliminar capas finales
            layersToRemove = {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'};
            lgraph = removeLayers(lgraph, layersToRemove);
            
            % Añadir nuevas capas para clasificación binaria
            newLayers = [
                fullyConnectedLayer(2, 'Name', 'fc_skinCancer', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
                softmaxLayer('Name', 'softmax_skinCancer')
                classificationLayer('Name', 'ClassificationLayer_skinCancer')
            ];
            lgraph = addLayers(lgraph, newLayers);
            lgraph = connectLayers(lgraph, 'avg_pool', 'fc_skinCancer');
            
            %% Balancear conjuntos de entrenamiento y validación
            labelCountTrain = countEachLabel(imdsTrain);
            minSetCountTrain = min(labelCountTrain.Count);
            imdsTrainBalanced = splitEachLabel(imdsTrain, minSetCountTrain, 'randomize');
            
            labelCountVal = countEachLabel(imdsVal);
            minSetCountVal = min(labelCountVal.Count);
            imdsValBalanced = splitEachLabel(imdsVal, minSetCountVal, 'randomize');
            
            %% Crear augmentedImageDatastores para redimensionar a [224 224]
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
            
            %% Guardar el modelo entrenado
            save('trainedSkinCancerResNet50.mat', 'net');
            fprintf('Modelo entrenado guardado en trainedSkinCancerResNet50.mat\n');
        
        
            fprintf('Modo TRAIN finalizado.\n');
            return;  % Finaliza la ejecución en modo train.
            
        case 'test'
            %% Modo test: cargar (o descargar) el modelo entrenado
            if ~isfile('trainedSkinCancerResNet50.mat')
                fprintf('No se encontró el modelo entrenado. Descargando desde GitHub release...\n');

                modelURL = 'https://github.com/JBLOZ/CNN-classification----geometric2d-and-cancer-dangerousness/releases/download/v(3.0.0)/trainedSkinCancerResNet50.mat';
                websave('trainedSkinCancerResNet50.mat', modelURL);
            end
            load('trainedSkinCancerResNet50.mat', 'net');
            fprintf('Modelo cargado.\n');


            if ~isfolder(testFolder)
            fprintf('La carpeta "test" no se encontró.');
            fprintf('Descargando skinTestData.zip desde GitHub release...\n');

            testDataURL = 'https://github.com/JBLOZ/CNN-classification----geometric2d-and-cancer-dangerousness/releases/download/v(3.0.0)/skinTestData.zip';
            websave('skinTestData.zip', testDataURL);
            unzip('skinTestData.zip');

            end
            fprintf('\nRealizando pruebas en 15 imágenes aleatorias del conjunto de test:\n');
            imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            inputSize = net.Layers(1).InputSize;
            numTestImages = 15;
            perm = randperm(numel(imdsTest.Files), numTestImages);
            for i = 1:numTestImages
                img = imread(imdsTest.Files{perm(i)});
                imgResized = imresize(img, inputSize(1:2));
                
                % Realizar la clasificación
                scores = predict(net, imgResized);
                maxScore = max(scores);
                if maxScore == scores(1)
                    predictedLabel = 'benign';
                else
                    predictedLabel = 'malignant';
                end
                
                trueLabel = imdsTest.Labels(perm(i));
                if strcmpi(char(trueLabel), predictedLabel)
                    status = 'OK';
                else
                    status = 'FAIL';
                end
                
                fprintf('Imagen %d: Etiqueta predicha: %s, Confianza: %.2f%%, %s\n', ...
                    i, predictedLabel, maxScore*100, status);
            end
            
            % Evaluación completa en el conjunto de test:
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
            
        otherwise
            error('Modo desconocido. Use ''train'' o ''test''.');
    end
    
    