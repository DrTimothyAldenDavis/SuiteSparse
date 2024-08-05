function [x,stats] = paru (A,b,opts)    %#ok
%PARU solve Ax=b using ParU sparse LU factorization
%
% Usage: x = paru(A,b), computes x=A\b using the ParU LU factorization method,
% a parallel UMFPACK-style method that factorizes multiple frontal matrices in
% parallel.
%
% [x,stats] = paru (A,b,opts)
%
% opts: an optional struct that controls ParU parameters:
%
%   opts.strategy:  ordering strategy, as a string (default: 'auto')
%      'auto': strategy is selected automatically
%      'symmetric': ordering of A+A', with preference for diagonal pivoting.
%           Works well for matrices with mostly symmetric nonzero pattern.
%      'unsymmetric': ordering A'*A, with no preference for diagonal pivoting.
%           Works well for matrices with unsymmetric nonzero pattern.
%
%   opts.tol:   relative pivot tolerance for off-diagonal entries (default:
%       0.1).  Pivot entries must be 0.1 * the max absolute value in its column.
%
%   opts.diagtol:   relative pivot tolerance for diagonal pivot entries when
%       using the symmetric strategy (default: 0.001).  A lower tolerance for
%       diagonal entries tends to reduce fill-in.
%
%   opts.ordering:  fill-reducing ordering option, as a string (default: 'amd')
%       'amd': AMD for the symmetric strategy, COLAMD for unsymmetric
%       'cholmod': use CHOLMOD's ordering strategy: try AMD or COLAMD,
%           and then try METIS if the fill-in from AMD/COLAMD is high;
%           then selects the best ordering found.
%       'metis': METIS on A+A' for symmetric strategy, A'*A for unsymmetric
%       'metis_guard': use the 'metis' ordering unless the matrix has one or
%           more rows with 3.2*sqrt(n) entries, in which case use 'amd'
%       'none': no fill-reducing ordering.
%
%   opts.prescale: prescale the input matrix (default: 'max')
%       'none': no prescaling
%       'sum': The prescaled matrix is R*A where R(i,i) = 1/sum(abs(A(i,:))).
%       'max': The prescaled matrix is R*A where R(i,i) = 1/max(abs(A(i,:))).
%
% stats: an optional output that provides information on the ParU
% analysis and factorization of the matrix:
%
%   stats.analysis_time: symbolic analysis time in seconds
%   stats.factorization_time: numeric factorization time in seconds
%   stats.solve_time: forward/backward solve time in seconds
%   stats.strategy_used: symmetric or unsymmetric
%   stats.ordering_used: amd(A+A'), colamd(A), metis(A+A'), metis(A'*A), or
%       none.
%   stats.flops: flop count for LU factorization
%   stats.lnz: # of entries in L
%   stats.unz: # of entries in U
%   stats.rcond: rough estimate of the recripical of the condition number
%   stats.blas: BLAS library used, as a string
%   stats.front_tree_tasking: a string stating how the paru mexFunction was
%       compiled, whether or not tasking is available for factorizing multiple
%       fronts at the same time ('sequential' or 'parallel').  Requires OpenMP
%       tasking.
%   stats.openmp: whether or not ParU is using OpenMP (a string).
%
% Note that if the matrix is singular, ParU will report an error, while
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
% See also paru_make, paru_demo, paru_many, paru_tiny, mldivide, amd, colamd.
%
% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

error ('paru mexFunction not yet compiled; see paru_make') ;

