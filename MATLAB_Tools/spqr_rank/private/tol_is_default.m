function s = tol_is_default (tol)
%TOL_IS_DEFAULT return true if tol is default, false otherwise
% usage: s = tol_is_default (tol)

% spqr_rank, Copyright (c) 2012, Leslie Foster and Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

s = (isempty (tol) || ischar (tol) || (isreal (tol) && tol < 0)) ;

