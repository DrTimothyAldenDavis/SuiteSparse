function b = rhs (m)
% b = rhs (m), compute a right-hand-side
% Example:
%   b = rhs (30) ;
% See also: cs_demo

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

b = ones (m,1) + (0:m-1)'/m ;
