function paru_demo
%PARU_DEMO test a single sparse matrix in ParU
%
% Usage: paru_demo
%
% See also paru, paru_make, paru_many, paru_tiny, mldivide.

% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

type paru_demo
load west0479 %#ok<LOAD>
A = west0479 ;
n = size (A,1) ;
b = rand (n,1) ;
x = A\b ;
norm (A*x-b)
[x2,paru_stats] = paru (A,b) ;
paru_stats %#ok<NOPRT>
norm (A*x2-b)

