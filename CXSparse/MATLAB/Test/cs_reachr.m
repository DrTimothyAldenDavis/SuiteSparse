function x = cs_reachr(L,b)                                                 %#ok
%CS_REACHR recursive reach (interface to CSparse cs_reachr)
% find nonzero pattern of x=L\sparse(b).  L must be sparse, real, and lower
% triangular.  b must be a real sparse vector.
%
% Example:
%   x = cs_reachr(L,b)
% See also: cs_demo

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+


error ('cs_reach mexFunction not found') ;

