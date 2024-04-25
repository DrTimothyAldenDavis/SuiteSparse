function [x,stats] = paru (A,b,opts)
%PARU solve Ax=b using ParU
%
% Usage: x = paru(A,b), computes x=A\b using ParU.
% Note that the MATLAB cannot use the fully parallel version of
% ParU, because of limitations in the MATLAB memory allocator for
% C/C++ mexFunctions.  The performance of the paru MATLAB
% interface will thus be slower than when using its C/C++
% interface.
%
% If the matrix is singular, ParU will report an error, while
% x=A\b reports a warning instead.
%
% Example:
%
%   load west0479
%   A = west0479 ;
%   n = size (A,1) ;
%   b = rand (n,1) ;
%   x1 = A\b ;
%   norm (A*x1-b)
%   x2 = paru (A,b) ;
%   norm (A*x2-b)
%
% [x,stats] = paru (A,b,opts)
%
% opts: an optional struct that sets the ParU parameters:
%   FIXME: describe opts input
% stats: an optional output that provides information on the ParU
% analysis and factorization of the matrix:
%   FIXME: describe stats output
%
% See also paru_make, paru_demo, paru_many, paru_tiny, mldivide.
%
% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

error ('paru mexFunction not yet compiled; see paru_make') ;

