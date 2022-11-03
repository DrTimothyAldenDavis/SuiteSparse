function L = cholup (Lold,w)
%CHOLUP Cholesky update, using Given's rotations
% given Lold and w, compute L so that L*L' = Lold*Lold' + w*w'
% Example:
%   L = cholup (Lold,w)
% See also: cs_demo

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

n = size (Lold,1) ;
L = [Lold w] ;

for k = 1:n

    g = givens (L(k,k), L(k,n+1)) ;

    L (:, [k n+1]) = L (:, [k n+1]) * g' ;

    disp ('L:') ;
    disp (L)
    pause
end

L = L (:,1:n) ;
disp (L) ;
