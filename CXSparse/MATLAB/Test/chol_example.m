function chol_example
%CHOL_EXAMPLE simple Cholesky factorization example
% Example
%   chol_example
% See also: testall

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

D = 10 ;
X = 1 ;
o = 0 ;

A = sparse ([
D o X o o o o X o o
o D o o X o o o o X
X o D o o o X o o o
o o o D o o o o X X
o X o o D o o o X X
o o o o o D X X o o
o o X o o X D o o o
X o o o o X o D X X
o o o X X o o X D o
o X o X X o o X o D ]) ;

disp ('A = ') ; disp (A) ;

L = chol(A)' ;

disp ('L = ') ; disp (L) ;

clf
subplot (1,2,1) ; spy (A) ;
subplot (1,2,2) ; spy (L+L') ;
