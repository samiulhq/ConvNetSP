rng(123);
clear all;
close all;
   %'RandRotation',[0 360], ...
augmenter = imageDataAugmenter( ...
     'RandXReflection',0.5,...
    'RandYReflection',0.5);

imageFolder = fullfile('Data','Training_Binary_Kenn');
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);


imds.ReadFcn = @customReadDatstoreImage;




tbl = countEachLabel(imds);
%[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
save('data','testSet','trainingSet');

augimds = augmentedImageDatastore([128 96 3],trainingSet,'DataAugmentation',augmenter);

layers = [
    imageInputLayer([128 96 3])
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.5)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.5)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer       
    dropoutLayer(0.5)
    fullyConnectedLayer(64)
    fullyConnectedLayer(64)
    fullyConnectedLayer(64)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


imageSize = layers(1).InputSize;

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'SquaredGradientDecayFactor',0.99,...
    'OutputFcn',@outfcn,...
    'MaxEpochs',200, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testSet, ...
    'ValidationFrequency',36, ...
    'Verbose',true, ...
    'MiniBatchSize',32, ...
    'Plots','training-progress');
net = trainNetwork(augimds,layers,options);

save('trained_convnet_kenn_binary_data','net');


