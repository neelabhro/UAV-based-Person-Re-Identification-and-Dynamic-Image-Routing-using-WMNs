%% This is a demo for the XQDA metric learning, as well as the evaluation on the VIPeR database. You can run this script to reproduce our CVPR 2015 results.
% Note: this demo requires about 1.0-1.4GB of memory.

close all; clear; clc;

feaFile1 = 'custom_probe2.mat';
feaFile2 = 'custom_gal2.mat';

numClass = 40;
numFolds = 10;
numRanks = 20;

%% load the extracted LOMO features
load(feaFile1, 'probe');
load(feaFile2, 'gallery');
galFea = gallery(:,1 : numClass);
probFea = probe(:,1 : numClass);
galFea = galFea';
probFea = probFea';
clear descriptors

%% set the seed of the random stream. The results reported in our CVPR 2015 paper are achieved by setting seed = 0. 
seed = 0;
rng(seed);

%% evaluate
cms = zeros(numFolds, numRanks);

for nf = 1 : numFolds
    p = randperm(numClass);
    
    galFea1 = galFea( p(1:numClass/2), : );
    probFea1 = probFea( p(1:numClass/2), : );
    
    t0 = tic;
    [W, M] = XQDA(galFea1, probFea1, (1:numClass/2)', (1:numClass/2)');

    %{
    %% if you need to set different parameters other than the defaults, set them accordingly
    options.lambda = 0.001;
    options.qdaDims = -1;
    options.verbose = true;
    [W, M] = XQDA(galFea1, probFea1, (1:numClass/2)', (1:numClass/2)', options);
    %}
    
    
    trainTime = toc(t0);
    galFea2 = galFea(p(numClass/2+1 : end), : );
    probFea2 = probFea(p(numClass/2+1 : end), : );
    

        
    t0 = tic;
    dist = MahDist(M, galFea2 * W, probFea2 * W);
    clear galFea2 probFea2 M W
    matchTime = toc(t0);
    
    fprintf('Fold %d: ', nf);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime);
    grid on;
    cms(nf,:) = EvalCMC( -dist,1 : numClass / 2, 1 : numClass / 2, numRanks );
    clear dist
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(nf,[1,5,10]) * 100);
end

meanCms = mean(cms);
plot(1 : numRanks, meanCms);

