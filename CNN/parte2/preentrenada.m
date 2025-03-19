zip = "ImagenesProyecto01.zip"; %Carpeta ZIP
carpeta_imgs = "ImagenesExtraidas";  % Carpeta donde se extraerán las imágenes

% Extraer el ZIP si no está extraído
if ~isfolder(carpeta_imgs)
    unzip(zip, carpeta_imgs);
end

nombre_img = "nevera.jpg"; %Imagen que vamos a analizar

ruta_img = fullfile(carpeta_imgs, nombre_img); %Se busca la imagen en la carpeta extraída

% Verificar si la imagen existe
if ~isfile(ruta_img)
    error("La imagen '%s' no se encontró en la carpeta extraída.", nombre_img);
end

I = imread(ruta_img); % Leer la imagen

% Cargar la red neuronal preentrenada
[red,nombres_clases] = imagePretrainedNetwork("googlenet");
tam = red.Layers(1).InputSize;

% Redimensionar la imagen para que coincida con la entrada de la red
X = imresize(I, tam(1:2));

% Convertir a un formato compatible con la red
X = single(X);
if canUseGPU
    X = gpuArray(X);
end

% Hacer la predicción
puntuaciones = predict(red, X);
[etiqueta, puntuacion] = scores2label(puntuaciones, nombres_clases);

% Ordenar las mejores 5 predicciones
[~, idx] = sort(puntuaciones, "descend");
idx = idx(5:-1:1);
mejores_clases = nombres_clases(idx);
mejores_puntuaciones = puntuaciones(idx);

% Figura 1: Mostrar la imagen con la predicción
figure;
imshow(I);
title(string(etiqueta) + ", " + num2str(gather(puntuacion)), 'FontWeight', 'bold');


% Figura 2: Mostrar gráfico de barras con el top 5 de predicciones
figure;
barh(mejores_puntuaciones);
xlim([0 1]);
title("Top 5 Predicciones", 'FontWeight', 'bold');
xlabel("Probabilidad");
yticklabels(mejores_clases);
