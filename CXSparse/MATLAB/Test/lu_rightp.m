function [L,U,P] = lu_rightp (A)
%LU_RIGHTP right-looking LU factorization, with partial pivoting.
%
% Example:
%   [L,U,P] = lu_rightp (A)
% See also: cs_demo

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

n = size (A,1) ;
P = eye (n) ;
for k = 1:n
    [x,i] = max (abs (A (k:n,k))) ;                           % partial pivoting
    i = i+k-1 ;
    P ([k i],:) = P ([i k], :) ;
    A ([k i],:) = A ([i k], :) ;                              % (6.10), (6.11)
    A (k+1:n,k) = A (k+1:n,k) / A (k,k) ;                               % (6.12)
    A (k+1:n,k+1:n) = A (k+1:n,k+1:n) - A (k+1:n,k) * A (k,k+1:n) ;     % (6.9)
end
L = tril (A,-1) + eye (n) ;
U = triu (A) ;
