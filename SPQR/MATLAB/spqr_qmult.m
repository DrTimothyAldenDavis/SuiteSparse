function Y = spqr_qmult (H,X,method)                                        %#ok
%SPQR_QMULT computes Q'*X, Q*X, X*Q', or X*Q with Q in Householder form.
% Usage: Y = spqr_qmult (Q,X,method)
%
%   method = 0: Y = Q'*X    default if 3rd input argument is not present.
%   method = 1: Y = Q*X 
%   method = 2: Y = X*Q'
%   method = 3: Y = X*Q
%
% Example:
%   These two examples both compute the min-norm solution to an
%   under determined system, but the latter is much more efficient:
%
%   [Q,R,E] = spqr(A') ; x = Q*(R'\(E'*b)) ;
%
%   [Q,R,E] = spqr(A',struct('Q','Householder')) ;
%   x = spqr_qmult(Q,R'\(E'*b),1) ;
%
% See also SPQR, SPQR_SOLVE, QR, MTIMES

% SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

help spqr_qmult
error ('spqr_qmult mexFunction not found') ;
