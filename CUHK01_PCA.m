%CUHK01 + LOMO
%Neelabhro Roy
%IIIT-Delhi
                                                                                                                                                            
clear;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
clc;                                                                                                            
close all;

feaFile = 'cuhk01_lomo.mat';
%pcaFile = 'CUHK01_LOMO_XQDA.mat';

%feaFile1 = 'custom_probe2.mat';
%feaFile2 = 'custom_gal2.mat';

numClass = 40;
numFolds = 20;
numRanks = 20;

%% load the extracted LOMO features
%load(feaFile1, 'probe');
%load(feaFile2, 'gallery');
galFea = gallery(:,1 : numClass);
probFea = probe(:,1 : numClass);
galFea = galFea';
probFea = probFea';
%p = randperm(numClass);

    galFea1 = galFea( (1:numClass/2),: );
    probFea1 = probFea( (1:numClass/2), : );
    galFea2 = galFea((numClass/2+1 : end), : );
    probFea2 = probFea((numClass/2+1 : end), : );

    TrainSet = zeros(40,26960);
    TrainSet(1:20 ,:) = galFea1;
    TrainSet(21: end,:) = probFea1;
    
    t0 = tic;

    trainTime = toc(t0);

    TestSet = zeros(40,26960);
    TestSet(1:20 ,:) = galFea2;
    TestSet(21: end,:) = probFea2;
    [X ,W] = matlabPCA(TrainSet',20);