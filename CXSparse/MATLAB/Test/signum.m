function s = signum (x)
%SIGNUM compute and display the sign of a column vector x
% Example
%   s = signum(x)
% See also: testall

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

s = ones (length (x),1) ;
s (find (x < 0)) = -1 ;     %#ok
disp ('s =') ;
disp (s) ;
