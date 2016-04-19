% Author : Cong Jie, Ng
% Paper  : Ng, C. J., and Teoh, A. B. J. "DCTNet: A Simple Learning-Free Approach for Face Recognition." In 2015 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), 761-68, 2015.

function OutImgs = DCTNet_Convolution(InImgs, Filters, Params, layer)
    
    OutImgs = {};
    numFilter = Params.NumFilters(layer);
    patchSize = Params.FilterSize(layer);
    filter = Filters{layer};
    mag = (patchSize-1)/2;
    
    for i = 1:length(InImgs)
        imgs = InImgs{i};
        for j = 1:length(imgs)
            [h, w] = size(imgs{j});
            img = zeros(h+patchSize-1,w+patchSize-1, 1);
            img((mag+1):end-mag,(mag+1):end-mag,:) = imgs{j};
            img = im2col(img, [patchSize patchSize]);
            
            if Params.FilterType == 1
                img = bsxfun(@minus, img, mean(img, 1));
            end
            
            filteredImgs = {};
            for p = 1:numFilter
                fImg = reshape(filter(:,p)'*img, [h w]);
                filteredImgs = cat(1, filteredImgs, fImg);
            end
            OutImgs = cat(1, OutImgs, { filteredImgs });
        end
    end
end