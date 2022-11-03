function r = gee_its_simple_resid (A, x, b)
%GEE_ITS_SIMPLE_RESID compute the relative residual for x=A\b
% For non-singular matrices with reasonable condition number, the relative
% residual is typically O(eps).
%
% Example:
%
%   r = gee_its_simple_resid (A, x, b) ;
%
% See also: norm, gee_its_simple

% GEE, Copyright (c) 2006-2007, Timothy A Davis. All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

r = norm (A*x-b,inf) / (norm (A,inf) * norm (x, inf) + norm (b, inf)) ;

