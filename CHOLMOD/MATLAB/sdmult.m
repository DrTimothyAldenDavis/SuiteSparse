function C = sdmult (S,F,transpose)                                   %#ok
%SDMULT sparse matrix times dense matrix
% Compute C = S*F or S'*F where S is sparse and F is full (C is also
% sparse).  S and F must both be real or both be complex.
%
%   Example:
%       C = sdmult (S,F) ;       C = S*F
%       C = sdmult (S,F,0) ;     C = S*F
%       C = sdmult (S,F,1) ;     C = S'*F
%
% See also mtimes.

 % Copyright 2006-2023, Timothy A. Davis, All Rights Reserved.
 % SPDX-License-Identifier: GPL-2.0+

error ('sdmult mexFunction not found') ;


