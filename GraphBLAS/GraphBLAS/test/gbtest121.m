function gbtest121
%GBTEST121 test times with scalars

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

a = pi ;
b = 2 ;
c1 = a.*b ;
c2 = GrB (a) .* GrB (b) ;

assert (isequal (c1, c2)) ;

fprintf ('gbtest121: all tests passed\n') ;

