%CSPL Paper Replication 65% Accuracy
%Neelabhro Roy
%IIIT-Delhi

%CUHK01 + LOMO
%Neelabhro Roy
%IIIT-Delhi
                                                                                                                                                            
clear;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
clc;                                                                                                            
close all;

%pcaFile = 'CUHK01_LOMO_XQDA.mat';

feaFile1 = 'custom_probe2.mat';
feaFile2 = 'custom_gal2.mat';
pcaFile = 'custom_PCA20.mat';

numClass = 40;
%numFolds = 20;
%numRanks = 20;

%% load the extracted LOMO features
load(feaFile1, 'probe');
load(feaFile2, 'gallery');
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
    
    Lu = 0.05*1;
    L = 0.0000000001;
    Lv = 0.2*1;
    La = 0.2*1;
    Lp = 0.2*1;
    Lw = 0.5*1;

    nu = 1*1;
    beta = 1*1;

    n = 20;
    d = 20;
    k = d;
    
    %[X , W] = matlabPCA(TrainSet',100);
    load(pcaFile, 'X');
    load(pcaFile, 'W');
    
    X2 = X(:, 1:20);
    X1 = X(:, 21:end);
    
    TestPCA = W' * TestSet';
    X22 = TestPCA(:, 1:20);
    X12 = TestPCA(:, 21:end);

    U  = randi([0, 1], [d,k]);
    V1 = randi([0, 1], [k,n]);
    V2 = randi([0, 1], [k,n]);
    A  = randi([0, 1], [k,k]);
    P1 = randi([0, 1], [k,d]);
    P2 = randi([0, 1], [k,d]);
    W2 = eye(k);
    



%% Main algorithm
    for i = 1:500
        U  = (( W2*X1 * transpose(V1)) + ( W2*X2 * transpose(V2)))/((( V1 * transpose(V1)) + ( V2 * transpose(V2)) + (Lu*eye(k))));
        V1 = (((transpose(U) * U) + (nu + beta + Lv) * eye(k))) \ ((transpose(U) *W2* X1) + (beta* A * V2) + nu * P1 * W2*X1);
        V2 = (((transpose(U) * U) + ( beta * transpose(A) * A) + (nu + Lv) .* eye(k))) \ ((transpose(U) *W2* X2) + (beta* transpose(A) * V1) + nu * P2*W2 * X2);
        P1 = (V1 * transpose(X1)) / ((X1 * transpose(X1)) + (Lp/nu)*eye(k));
        P2 = (V2 * transpose(X2)) / ((X2 * transpose(X2)) + (Lp/nu)*eye(k));
        A  = (V1 * transpose(V2)) / ((V2 * transpose(V2)) + (La/beta)*eye(k));
        
    end

    
    D = zeros(n,n);
    for m = 1:n
    
        v1 = P1*W2*(X12(:,m));
    
        for i = 1:n
            v2 = P2*W2*(X22(:,i));
            D(m,i) = norm(((v1 - A*v2)));
        end
        
    end
    
CMC(D,20);
hold on;




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
plot(1 : numRanks, meanCms*100,'LineWidth',3);
hold on;
legend('CSPL','XQDA');