function C = sptranspose (A,conj)                                           %#ok
%SPTRANSPOSE: compute the transpose, or conjugate-transpose, of a sparse matrix.
% This function is faster and more memory-efficient than the built-in sparse
% transpose and ctranspose, at least in versions of MATLAB prior to 7.7 or so,
% when the MATLAB built-in operator was upgraded by The MathWorks, Inc.
%
% Example:
%       C = sptranspose (A) ;       % computes C = A.'
%       C = sptranspose (A,1) ;     % computes C = A'
%
% See also transpose, ctranspose.

% SSMULT, Copyright (c) 2007-2011, Timothy A Davis. All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

error ('sptranspose mexFunction not found') ;

