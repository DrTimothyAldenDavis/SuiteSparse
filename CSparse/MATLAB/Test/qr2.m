function [V,Beta,R] = qr2 (A)
%QR2 QR factorization based on Householder reflections
%
% Example:
%   [V,beta,R] = qr2 (A)
% See also: testall

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

[m n] = size (A) ;
V = zeros (m,n) ;
Beta = zeros (1,n) ;
for k = 1:n
    % [v,beta,s] = gallery ('house', A (k:m,k), 2) ;
    [v,beta] = house (A (k:m,k)) ;
    V (k:m,k) = v ;
    Beta (k) = beta ;
    A (k:m,k:n) = A (k:m,k:n) - v * (beta * (v' * A (k:m,k:n))) ;
end
R = triu (A) ;
