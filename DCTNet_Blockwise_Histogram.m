% Author : Cong Jie, Ng
% Paper  : Ng, C. J., and Teoh, A. B. J. "DCTNet: A Simple Learning-Free Approach for Face Recognition." In 2015 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), 761-68, 2015.

function OutHists = DCTNet_Blockwise_Histogram(InHashedImgs, Params)
    
    OutHists = {};
    blkSize = Params.HistBlockSize;
    
    for i = 1:length(InHashedImgs)
        hashedImg = InHashedImgs{i};
        
        % Crop HashedImg to center, in case HashedImg size is not divisible by BlockSize
        [h, w] = size(hashedImg);
        margin = [h w] - blkSize .* floor([h w] ./ blkSize);
        margin1 = round(margin / 2);
        margin2 = margin - margin1;
        hashedImg = hashedImg((margin1(1) + 1):(end - margin2(1)),(margin1(2) + 1):(end - margin2(2))); 
        
        % Block-Wise Histogramming 
        blockwiseHist = histc(im2col(hashedImg, blkSize, 'distinct'),(0:2^Params.NumFilters(end)-1)');
        OutHists = cat(1, OutHists, blockwiseHist);
    end
end