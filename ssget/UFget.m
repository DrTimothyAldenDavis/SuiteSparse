function Problem = UFget (matrix, index)
%UFGET former interface to the UF (now SuiteSparse) Matrix Collection
% This function works but is deprecated.  Use ssget instead.

% Copyright (c) 2009-2019, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

warning ('UFget:deprecated', 'UFget is deprecated; use ssget instead') ;
if (nargin == 0)
    Problem = ssget ;
elseif (nargin == 1)
    Problem = ssget (matrix) ;
elseif (nargin == 2)
    Problem = ssget (matrix, index) ;
else
    error ('usage: Problem = ssget (matrix, index)') ;
end

