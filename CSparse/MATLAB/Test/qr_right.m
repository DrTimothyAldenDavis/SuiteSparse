function [V,Beta,R] = qr_right (A)
%QR_RIGHT right-looking Householder QR factorization.
% Example:
%   [V,Beta,R] = qr_right (A)
% See also: cs_demo

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

[m n] = size (A) ;
V = zeros (m,n) ;
Beta = zeros (1,n) ;
for k = 1:n
    [v,beta] = gallery ('house', A (k:m,k), 2) ;
    V (k:m,k) = v ;
    Beta (k) = beta ;
    A (k:m,k:n) = A (k:m,k:n) - v * (beta * (v' * A (k:m,k:n))) ;
end
R = A ;
