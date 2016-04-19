% Author : Cong Jie, Ng
% Paper  : Ng, C. J., and Teoh, A. B. J. "DCTNet: A Simple Learning-Free Approach for Face Recognition." In 2015 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), 761-68, 2015.

function OutImgs = DCTNet_BinaryHashing(InImgs, Params)
    
    OutImgs = {};
    mapWeights = 2.^((Params.NumFilters(end)-1):-1:0);
    
    for d = 1:length(InImgs)
        imgs = InImgs{d};
        hashedImg = 0;
        for i = 1:length(imgs)
            hashedImg = hashedImg + mapWeights(i) * Heaviside(imgs{i});
        end
        OutImgs = cat(1, OutImgs, hashedImg);
    end
end

function X = Heaviside(X)
    X = sign(X);
    X(X <= 0) = 0;
end