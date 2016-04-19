% Author : Cong Jie, Ng
% Paper  : Ng, C. J., and Teoh, A. B. J. "DCTNet: A Simple Learning-Free Approach for Face Recognition." In 2015 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), 761-68, 2015.

function projMat = WPCA_Svd(X, dimension)

   rglEpsilon = 0.001;
   
   X = bsxfun(@minus, X, mean(X, 2));
   [eigVectors_PCA, eigValues_PCA, ~] = svd(X, 0); 
  
   eigVectors_PCA = eigVectors_PCA(:, 1 : dimension);
   eigVectors_PCA = eigVectors_PCA';
   assert(size(eigVectors_PCA, 1) == dimension);
  
   eigValues_PCA = diag(eigValues_PCA.^2);
   eigValues_PCA = eigValues_PCA(1 : dimension);
  
   % Whitening PCA
   eigValues_wPCA = diag(1./sqrt(eigValues_PCA + rglEpsilon));
   projMat = eigValues_wPCA * eigVectors_PCA;

end