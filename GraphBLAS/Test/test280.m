function test280
%TEST280 subassign method 26

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

load west0479
GB_mex_grow (west0479) ;

n = 10000 ;
nz = 10e6 ;
A = sprand (n, n, nz/n^2) ;
GB_mex_grow (A) ;

n = 2e6 ;
nz = 10e6 ;
A = sprand (n, n, nz/n^2) ;
GB_mex_grow (A) ;

fprintf ('test280 all tests passed.\n') ;

