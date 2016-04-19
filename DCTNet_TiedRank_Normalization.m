% Author : Cong Jie, Ng
% Paper  : Ng, C. J., and Teoh, A. B. J. "DCTNet: A Simple Learning-Free Approach for Face Recognition." In 2015 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), 761-68, 2015.

function OutTRHists = DCTNet_TiedRank_Normalization(InBlkHists)
    
    OutTRHists = {};
    
    for i = 1:length(InBlkHists)
        trHists = [];
        blkHists = InBlkHists{i};
        for j = 1:size(blkHists, 2)
            % Rank without zero
            trHist = zeros(size(blkHists(:,j)));
            trHist(blkHists(:,j) ~= 0) = tiedrank(nonzeros(blkHists(:,j)));
            
            % Feature Evenization 
            trHist = sqrt(trHist);
            trHist = trHist / norm(trHist);
            trHists = cat(2, trHists, trHist);
        end
        OutTRHists = cat(1, OutTRHists, trHists);
    end
end