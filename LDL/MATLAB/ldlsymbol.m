function [Lnz, Parent, fl] = ldlsymbol (A, P)				    %#ok
%LDLSYMBOL symbolic Cholesky factorization
%
% Example:
%       [Lnz, Parent, fl] = ldlsymbol (A)
%       [Lnz, Parent, fl] = ldlsymbol (A, P)
%
% P is a permutation of 1:n, an output of AMD, SYMAMD, or SYMRCM, for example.
% Only the diagonal and upper triangular part of A or A(P,P) is accessed; the
% lower triangular part is ignored.  If P is not provided, then P = 1:n is
% assumed.
%
% The elimination tree is returned in the Parent array.  The number of nonzeros
% in each column of L is returned in Lnz.  fl is the floating point operation
% count for a subsequent LDL' factorization.  This mexFunction replicates the
% following MATLAB computations, using ldl_symbolic:
%
%       Lnz = symbfact (A) - 1 ;
%       Parent = etree (A) ;
%       fl = sum (Lnz .* (Lnz + 2)) ;
%
% or, if P is provided,
%
%       Lnz = symbfact (A (P,P)) - 1 ;
%       Parent = etree (A (P,P)) ;
%       fl = sum (Lnz .* (Lnz + 2)) ;
%
% Note that this routine is not required by LDL, since LDL does its own
% symbolic factorization.
%
% See also ldlsparse, symbfact, etree

% LDL, Copyright (c) 2005-2022 by Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

error ('ldlsymbol mexFunction not found') ;
