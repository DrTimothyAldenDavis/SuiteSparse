function resid = paru_tiny
%PARU_TINY test a tiny sparse matrix with ParU
%
% Usage: paru_tiny
%
% See also paru, paru_make, paru_demo, paru_many, mldivide.
%
% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

type paru_tiny
A = sparse (rand (2)) ;
b = rand (2,1) ;
x = paru (A,b) %#ok<*NOPRT>
resid = A*x-b

