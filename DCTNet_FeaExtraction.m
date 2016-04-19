% Author : Cong Jie, Ng
% Paper  : Ng, C. J., and Teoh, A. B. J. "DCTNet: A Simple Learning-Free Approach for Face Recognition." In 2015 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), 761-68, 2015.

function feature = DCTNet_FeaExtraction(InImg, Filters, Params)

    %% Represent Multi-Channel image with cell
    filteredImgs = {};
    for i = 1:length(InImg)
        filteredImgs = cat(1, filteredImgs, {InImg(i)});
    end
    
    %% Convolution Layers
    for layer = 1:Params.NumLayers
        filteredImgs = DCTNet_Convolution(filteredImgs, Filters, Params, layer);
    end

    %% Binary Hashing
    hashedImgs = DCTNet_BinaryHashing(filteredImgs, Params);
    
    %% Block-wise Histogram
    blockwiseHists = DCTNet_Blockwise_Histogram(hashedImgs, Params);
    
    %% Tied Rank Normalization
    if (Params.TiedRankNormalization)
        blockwiseHists = DCTNet_TiedRank_Normalization(blockwiseHists);
    end
    
    %% Vectorize to get final feature vector
    feature = cell2mat(blockwiseHists);
    feature = feature(:);
end
