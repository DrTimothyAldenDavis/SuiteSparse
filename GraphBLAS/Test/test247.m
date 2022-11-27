function test247
%TEST247: test saxpy3 fine-hash method

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 1000000 ;
A = GrB (n, n) ;
B = GrB (GrB.random (n, 1, 0.01), 'sparse') ;
A (1:100, 1:100) = GrB.random (100, 100, 0.4) ;
A = GrB (A, 'sparse') ;

nth = GrB.threads ;
chk = GrB.chunk ;
desc.axb = 'hash' ;
GrB.threads (16) ;
GrB.chunk (1) ;
GrB.burble (1) ;
C1 = GrB.mxm ('+.*', A, B, desc) ;
GrB.burble (0) ;

A = double (A) ;
B = double (B) ;
C2 = A*B ;
err = norm (C1 - C2, 1) ;
fprintf ('err: %g\n', err) ;
assert (err < 1e-12) ;

GrB.threads (nth) ;
GrB.chunk (chk) ;

fprintf ('\ntest247: all tests passed\n') ;

