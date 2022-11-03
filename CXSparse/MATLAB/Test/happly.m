function hx = happly (v, beta, x)
%HAPPLY apply Householder reflection to a vector
% Example:
%   hx = happly (v,beta,x) ;        % computes hx = x - v * (beta * (v' *x)) ;
% See also: testall

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

hx = x - v * (beta * (v' *x)) ;
