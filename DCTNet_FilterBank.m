% Author : Cong Jie, Ng
% Paper  : Ng, C. J., and Teoh, A. B. J. "DCTNet: A Simple Learning-Free Approach for Face Recognition." In 2015 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), 761-68, 2015.

function dctFilters = DCTNet_FilterBank(Params)
    dctFilters = cell(Params.NumLayers, 1);
    
    for i = 1:Params.NumLayers
        ps = Params.FilterSize(i);
        nf = Params.NumFilters(i);
        dctMatrix = DCT_Matrix(ps);
        diagIndex = AntiDiagonal_Index(ps, 1);
        dctMatrix(:, diagIndex) = dctMatrix(:,:);
        dctFilters{i} = dctMatrix(:, 2:nf+1);
    end
end

function dctMatrix = DCT_Matrix(N)
    dctMat = dctmtx(N);
    dctMatrix = zeros(N * N, N * N);

    for i = 1:N
        for j = 1:N
            M = dctMat(i,:)' * dctMat(j,:);
            dctMatrix(:, (i-1) * N + j) = M(:);
        end
    end
end

function indices = AntiDiagonal_Index(n, direction)
    % direction = 1 => north east
    % direction = 0 => south west
    indices = zeros(n, n);
    
    idx = 1;
    for i = 1:n
        u = i; v = 1;
        while u
            indices(u, v) = idx;
            indices(n - u + 1, n - v + 1) = n*n - idx + 1;
            idx = idx + 1;
            u = u - 1;
            v = v + 1;
        end
    end
    
    if direction
        indices = indices';
    end
end