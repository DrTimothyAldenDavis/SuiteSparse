function r = spex_rank (A)
% spex_rank: compute the exact rank of A.
% A is a sparse matrix of any size. The algorithm computes the exact rank
% of A via either left-looking LU factorization (if A is square) or exact
% QR factorization (if A is rectangular).\
%
% Usage:
%
% r = spex_rank (A) returns the rank of A
%
% See also vpa, spex_mex_install, spex_mex_test, spex_mex_demo,
%   spex_lu_backslash

% SPEX: (c) 2022, Chris Lourenco, Jinhao Chen,
% Lorena Mejia Domenzain, Timothy A. Davis and Erick Moreno-Centeno.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

if (~isnumeric (A))
    error ('inputs must be numeric') ;
end

% Check if the input matrix is stored as sparse. If not, SPEX expects
% sparse input, so convert to sparse.
if (~issparse (A))
    A = sparse (A) ;
end

% Preprocessing complete. Now use SPEX QR to solve A*x=b.
r=spex_rank_mex (A) ;

end

