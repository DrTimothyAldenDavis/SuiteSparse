function [x, info] = klus (A, b)
% [x, Info] = klus (A, b)
%
% Info (1): # of blocks
% Info (2): dimension of largest block
% Info (3): nnz (L+U+off diagonal blocks)

% this is out of date:
% usage: [x, Info] = klus (A, b, tol)
%
% tol: partial pivoting tolerance.  Use 1e-3 (for example) to prefer diagonal
% pivoting.

% THis is out of date:
% Info (1): n
% Info (2): nz in off diagonal part
% Info (3): # of blocks
% Info (4): max nz in diagonal blocks of A
% Info (5): dimension of largest block
% Info (6): estimated nz in L, incl. diagonal, excludes off-diagonal entries
% Info (7): estimated nz in U, incl. diagonal, excludes off-diagonal entries
%
% Info (8): nz in L, including diagonal, excludes off-diagonal entries
% Info (9): nz in U, including diagonal, excludes off-diagonal entries
% Info (10): analyze cputime
% Info (11): factor cputime
% Info (12): solve cputime
% Info (13): refactorize cputime (if computed)
% Info (14): # off-diagonal pivots chosen
%
% b may be n-by-m with m > 1.  It must be dense.
%

help klus
error ('klus mexFunction not found') ;
