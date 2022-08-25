function C = gb_expand (scalar, S, type)
%GB_EXPAND expand a scalar into a GraphBLAS matrix.
% Implements C = GrB.expand (scalar, S, type).  This function assumes the
% first input is a scalar; the caller has checked this already.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

% typecast the scalar to the desired type, and make sure it's full
scalar = gbfull (gbnew (scalar, type)) ;

% expand the scalar into the pattern of S
C = gbapply2 (['2nd.' type], S, scalar) ;

