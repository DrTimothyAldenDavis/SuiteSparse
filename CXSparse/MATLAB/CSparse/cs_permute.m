function C = cs_permute (A,p,q)                                             %#ok
%CS_PERMUTE permute a sparse matrix.
%   C = cs_permute(A,p,q) computes C = A(p,q)
%
%   Example:
%       Prob = ssget ('HB/arc130') ; A = Prob.A ; [m n] = size (A) ;
%       p = randperm (m) ; q = randperm (n) ;
%       C = cs_permute (A,p,q) ;    % C = A(p,q)
%
%   See also CS_SYMPERM, SUBSREF.

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

error ('cs_permute mexFunction not found') ;
