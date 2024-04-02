% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

clear all
% paru_make
A = sparse (rand (2)) ;
b = rand (2,1) ;
paru (A,b)
