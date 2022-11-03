function A = cs_sparse (i,j,x)                                              %#ok
%CS_SPARSE convert a triplet form into a sparse matrix.
%   A = cs_sparse(i,j,x) is identical to A = sparse(i,j,x), except that x must
%   be real, and the length of i, j, and x must be the same.
%
%   Example:
%       Prob = ssget ('HB/arc130') ; S = Prob.A ;
%       [i j x] = find (S) ;
%       A = cs_sparse (i,j,x) ;
%       S-A
%
%   See also FIND, SPARSE, SPCONVERT.

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

error ('cs_sparse mexFunction not found') ;
