function paru_many
%PARU_MANY: test many matrices with ParU
%
% Usage: paru_many
%
% See also paru, paru_make, paru_demo, paru_tiny, mldivide.

% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

index = ssget ;
square = find (index.nrows == index.ncols & index.isReal & ~index.cholcand) ;
nz = index.nnz (square) ;
[~,p] = sort (nz) ;
square = square (p) ;
nmat = length (square) ;

fprintf ('testing %d matrices:\n', nmat) ;

rng ('default') ;

for k = 1:nmat
    id = square (k) ;
    fprintf ('Matrix: %s/%s nz %d\n', index.Group {id}, index.Name {id}, index.nnz (id)) ;

    % get the problem
    Prob = ssget (id, index) ;
    A = Prob.A ;
    clear Prob
    if (~isreal (A))
        error ('!') ;
    end
    n = size (A,1) ;
    xtrue = rand (n,1) ;
    b = A*xtrue ;
    anorm = norm (A, 1) ;

    % try x=A\b
    lastwarn ('') ;
    x = A\b ;
    [lastmsg, lastid] = lastwarn ;
    lastwarn ('') ;
    if (isempty (lastid))
        resid = norm (A*x-b,1) / anorm ;
        fprintf ('A\\b  resid %g\n', resid) ;
        x2 = paru (A,b) ;
        resid2 = norm (A*x2-b,1) / anorm ;
        fprintf ('ParU resid %g\n', resid2) ;
    end
end

