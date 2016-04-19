% Author : Cong Jie, Ng
% Paper  : Ng, C. J., and Teoh, A. B. J. "DCTNet: A Simple Learning-Free Approach for Face Recognition." In 2015 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), 761-68, 2015.

clear all;

%% Parameters
Params.FilterType = 0;            % 0 => DCT, 1 => PCA filter (Shared by PCANet author) 
Params.TiedRankNormalization = 1; % 0 => Disable, 1 => Enable
Params.WPCA = 0;                  % 0 => Disable, 1 => Enable
Params.WPCADim = 300;             % WPCA = 0 will render WPCADim no effect 
Params.HistBlockSize = [16 16];   % Block-Wise Histogram Size 

% The following parameters has no effect for FilterType = 1 
Params.NumLayers = 2;             % Number of Convolution Layers
Params.FilterSize = [5 5];        % Filter Size
Params.NumFilters = [8 8];        % Number of Filters for each layer

%% Initialize Filters
if Params.FilterType == 0
    Filters = DCTNet_FilterBank(Params);
elseif Params.FilterType == 1
    load('PCANet_2_filters_MultiPIE_FaceRecog');
    Filters = V;
    Params.NumLayers = 2;
    Params.FilterSize = [5 5];
    Params.NumFilters = [8 8];
end

%% Load Face Data
load('FERET_B_64x64_(a,c,h,j,k)');

% Gallery ba, bj, bk (frontal faces)
trainData.X = [FERET_ba.X FERET_bj.X FERET_bk.X];
trainData.y = [FERET_ba.y FERET_bj.y FERET_bk.y];

% Probe bc, bh (+40, -40) degree in pose
testData = {FERET_bc, FERET_bh};
datasetNames = {'bc', 'bh'};

fprintf('\n ====== Parameters ======= \n')
disp(Params)

%% Feature Extraction
fprintf('\n ====== Gallery Feature Extraction ======= \n')
tic;

ftrain = [];
for i = 1:1:length(trainData.y)
    imgCell = { reshape(trainData.X(:, i), [imgHeight imgWidth]) };
    ftrain = cat(2, ftrain, DCTNet_FeaExtraction(imgCell, Filters, Params));
end
ftrain = ftrain';

fprintf('\n     DCTNet Gallery Feature Extraction Time : %.2f secs.\n', toc);

%% Dimension Reduction
if Params.WPCA ~= 0 
    meanTrain = mean(ftrain, 1);
    ftrain = bsxfun(@minus, ftrain, meanTrain);
    reduceMat = WPCA_Svd(ftrain', Params.WPCADim)';
    ftrain = ftrain * reduceMat;
end

%% Testing 
for i = 1:length(testData)
    tic
    ftest = [];
    for j = 1:length(testData{i}.y)
        imgCell = { reshape(testData{i}.X(:, j), [imgHeight imgWidth]) };
        ftest = cat(2, ftest, DCTNet_FeaExtraction(imgCell, Filters, Params));
    end
    FeaExtTime = toc;
    ftest = ftest';
    
    %% Dimension Reduction
    if Params.WPCA ~= 0 
        ftest = bsxfun(@minus, ftest, meanTrain);
        ftest = ftest * reduceMat;
    end

    %% Recognition Rate
    pairDist = pdist2(ftrain, ftest, 'cosine');
    [~,minIDX] = min(pairDist);
    outCRR = sum(testData{i}.y ==  trainData.y(minIDX))/length(testData{i}.y);

    %% Results display
    fprintf('\n ===== Results of DCTNet with NN classifier =====');
    fprintf('\n     Dataset %s', datasetNames{i});
    fprintf('\n     Feature Extraction Time: %.2f sec', FeaExtTime);
    fprintf('\n     Recognition Rate: %.6f', outCRR);
    fprintf('\n');
    
end