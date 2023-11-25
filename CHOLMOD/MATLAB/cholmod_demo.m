function cholmod_demo
%CHOLMOD_DEMO a demo for CHOLMOD
%
% Tests CHOLMOD with the sparse matrix problem used in the MATLAB bench
% program, with various sizes.  Note that MATLAB uses CHOLMOD itself for
% x=A\b, chol, etc. so the timings should be comparable.
%
% See CHOLMOD/MATLAB/Test/cholmod_test.m for a lengthy test using
% matrices from the SuiteSparse Matrix Collection.
%
% Example:
%   cholmod_demo
%
% See also bench.

 % Copyright 2006-2023, Timothy A. Davis, All Rights Reserved.
 % SPDX-License-Identifier: GPL-2.0+

help cholmod_demo

 % matrix from bench (n = 600 is used in 'bench'):
for n = [600 1200]
    A = delsq (numgrid ('L', n)) ;
    try_matrix (A) ;
end

 %-------------------------------------------------------------------------
function try_matrix (A)
 % try_matrix: try a matrix with CHOLMOD

n = size (A,1) ;
S = sparse (A) ;

fprintf ('\n--------------------------------------------------------------\n') ;
if (issparse (A))
    fprintf ('cholmod_demo: sparse matrix, n %d nnz %d\n', n, nnz (A)) ;
else
    fprintf ('cholmod_demo: dense matrix, n %d\n', n) ;
end

k = max (1,fix(n/2))  ;
C = A (:,k) * 0.1 ;
try
    % use built-in AMD
    p = amd (S) ;
catch
    try
        % use AMD from SuiteSparse (../../AMD)
        p = amd2 (S) ;
    catch
        % use SYMAMD
        p = symamd (S) ;
    end
end
S = S (p,p) ;

lnz = symbfact2 (S) ;
fl = sum (lnz.^2) ;

tic
L = lchol (S) ;             %#ok
t1 = toc ;
fprintf ('CHOLMOD lchol(sparse(A))       time: %6.2f    gflop %8.2f\n', ...
    t1, 1e-9 * fl / t1) ;

tic
LD = ldlchol (S) ;
t2 = toc ;
fprintf ('CHOLMOD ldlchol(sparse(A))     time: %6.2f    gflop %8.2f\n', ...
    t2, 1e-9 * fl / t2) ;

tic
LD2 = ldlupdate (LD,C) ;
t3 = toc ;
fprintf ('CHOLMOD ldlupdate(sparse(A),C) time: %6.2f (rank-1, C dense)\n', t3) ;

[L,D] = ldlsplit (LD2) ;
 % err = norm ((S+C*C') - L*D*L', 1) / norm (S,1) ;
err = ldl_normest ((S+C*C'), L, D) / norm (S,1) ;
fprintf ('err: %g\n', err) ;

tic
LD3 = ldlrowmod (LD, k) ;
t4 = toc ;
fprintf ('CHOLMOD ldlrowmod(LD,k)        time: %6.2f\n', t4) ;

[L,D] = ldlsplit (LD3) ;
S2 = S ;
I = speye (n) ;
S2 (k,:) = I (k,:) ;
S2 (:,k) = I (:,k) ;
 % err = norm (S2 - L*D*L', 1) / norm (S,1) ;
err = ldl_normest (S2, L, D) / norm (S,1) ;
fprintf ('err: %g\n', err) ;

LD4 = ldlchol (S2) ;
[L,D] = ldlsplit (LD4) ;
 % err = norm (S2 - L*D*L', 1) / norm (S,1) ;
err = ldl_normest (S2, L, D) / norm (S,1) ;
fprintf ('err: %g\n', err) ;

tic
R = chol (S) ;              %#ok
s1 = toc ;
fprintf ('MATLAB  chol(sparse(A))        time: %6.2f    gflop %8.2f\n', ...
    s1, 1e-9 * fl / s1) ;

fprintf ('CHOLMOD lchol(sparse(A)) speedup over chol(sparse(A)): %6.1f\n', ...
    s1 / t1) ;

b = sum (A)' ;

tic ;
x = A\b ;
t1 = toc ;
e1 = norm (A*x-b, 1) ;

tic ;
x = cholmod2 (A,b) ;
t2 = toc ;
e2 = norm (A*x-b) ;

fprintf ('MATLAB  x=A\\b      time: %8.4f  resid: %8.0e\n', t1, e1) ;
fprintf ('CHOLMOD x=A\\b      time: %8.4f  resid: %8.0e\n', t2, e2) ;
fprintf ('CHOLMOD speedup: %8.2f\n', t1/t2) ;

if (n > 4000)
    % problem is too large for full matrix tests
    return ;
end

 % tests with full matrices:
X = full (C) ;
E = full (A) ;
tic
R = chol (E) ;
s2 = toc ;
fprintf ('MATLAB  chol(full(A))          time: %6.2f    gflop %8.2f\n', ...
    s2, 1e-9 * fl / s2) ;

Z = full (R) ;
tic
Z = cholupdate (Z,X) ;
s3 = toc ;
fprintf ('MATLAB  cholupdate(full(A),C)  time: %6.2f (rank-1)\n', s3) ;

err = norm ((E+X*X') - Z'*Z, 1) / norm (E,1) ;
fprintf ('err: %g\n', err) ;

fprintf ('CHOLMOD sparse update speedup vs MATLAB DENSE update:  %6.1f\n', ...
    s3 / t3) ;

