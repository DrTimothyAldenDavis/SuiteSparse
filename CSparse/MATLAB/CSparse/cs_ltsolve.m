function x = cs_ltsolve (L,b)                                               %#ok
%CS_LTSOLVE solve a sparse upper triangular system L'*x=b.
%   x = cs_ltsolve(L,b) computes x = L'\b, L must be lower triangular with a
%   zero-free diagonal.  b must be a full vector.
%
%   Example:
%       Prob = ssget ('HB/bcsstk01') ; L = cs_chol (Prob.A) ; n = size (L,1) ;
%       b = rand (n,1) ; x = cs_ltsolve (L,b) ; norm (L'*x-b)
%
%   See also CS_LSOLVE, CS_USOLVE, CS_UTSOLVE, MLDIVIDE.

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

error ('cs_ltsolve mexFunction not found') ;
