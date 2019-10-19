function cholmod_demo
%CHOLMOD_DEMO: a demo for CHOLMOD
%
% Tests CHOLMOD with various randomly-generated matrices, and the west0479
% matrix distributed with MATLAB.  Random matrices are not good test cases,
% but they are easily generated.  It also compares CHOLMOD and MATLAB on the
% sparse matrix problem used in the MATLAB BENCH command.
%
% See CHOLMOD/MATLAB/Test/test_all.m for a lengthy test using matrices from
% the UF sparse matrix collection.
%
% See also BENCH

%   Copyright 2006, Timothy A. Davis
%   http://www.cise.ufl.edu/research/sparse

help cholmod_demo

rand ('state', 0) ;
randn ('state', 0) ;

load west0479
A = west0479 ;
n = size (A,1) ;
A = A*A'+100*speye (n) ;
try_matrix (A) ;
clear A

n = 2000 ;
A = sprandn (n, n, 0.002) ;
A = A+A'+100*speye (n) ;
try_matrix (A) ;
clear A

for n = [100 2000]
    A = rand (n) ;
    A = A*A' + 10 * eye (n) ;
    try_matrix (A) ;
    clear A
end

fprintf ('\n--------------------------------------------------------------\n') ;
fprintf ('\nWith the matrix used in the MATLAB 7.2 "bench" program.\n') ;
fprintf ('No fill-reducing orderings are used; type "help bench" for more') ;
fprintf (' information.\n') ;

n = 300 ;
A = delsq (numgrid ('L', n)) ;
b = sum (A)' ;

spparms ('default') ;
spparms ('autommd',0) ;
spparms ('autoamd',0) ;
tic ;
x = A\b ;
t1 = toc ;
e1 = norm (A*x-b) ;

tic ;
x = cholmod (A,b,0) ; 
t2 = toc ;
e2 = norm (A*x-b) ;

fprintf ('MATLAB  x=A\\b      time: %8.4f  resid: %8.0e\n', t1, e1) ;
fprintf ('CHOLMOD x=A\\b      time: %8.4f  resid: %8.0e\n', t2, e2) ;
fprintf ('CHOLMOD speedup: %8.2f\n', t1/t2) ;

spparms ('default') ;

fprintf ('\ncholmod_demo finished: all tests passed\n') ;
fprintf ('\nFor more accurate timings, run this test again.\n') ;






function try_matrix (A)

n = size (A,1) ;
S = sparse (A) ;

fprintf ('\n--------------------------------------------------------------\n') ;
if (issparse (A))
    fprintf ('cholmod_demo: sparse matrix, n %d nnz %d\n', n, nnz (A)) ;
else
    fprintf ('cholmod_demo: dense matrix, n %d\n', n) ;
end

X = rand (n,1) ;
C = sparse (X) ;
try
    p = amd (S) ;
catch
    p = symamd (S) ;
end
S = S (p,p) ;

lnz = symbfact2 (S) ;
fl = sum (lnz.^2) ;

tic
L = lchol (S) ;
t1 = toc ;
fprintf ('CHOLMOD lchol(sparse(A))       time: %6.2f    mflop %8.1f\n', ...
    t1, 1e-6 * fl / t1) ;

tic
LD = ldlchol (S) ;
t2 = toc ;
fprintf ('CHOLMOD ldlchol(sparse(A))     time: %6.2f    mflop %8.1f\n', ...
    t2, 1e-6 * fl / t2) ;

tic
LD = ldlupdate (LD,C) ;
t3 = toc ;
fprintf ('CHOLMOD ldlupdate(sparse(A),C) time: %6.2f (rank-1, C dense)\n', t3) ;

[L,D] = ldlsplit (LD) ;
L = full (L) ;
err = norm ((S+C*C') - L*D*L', 1) / norm (S,1)

tic
R = chol (S) ;
s1 = toc ;
fprintf ('MATLAB  chol(sparse(A))        time: %6.2f    mflop %8.1f\n', ...
    s1, 1e-6 * fl / s1) ;

E = full (A) ;
tic
R = chol (E) ;
s2 = toc ;
fprintf ('MATLAB  chol(full(A))          time: %6.2f    mflop %8.1f\n', ...
    s2, 1e-6 * fl / s2) ;

Z = full (R) ;
tic
Z = cholupdate (Z,X) ;
s3 = toc ;
fprintf ('MATLAB  cholupdate(full(A),C)  time: %6.2f (rank-1)\n', s3) ;

err = norm ((E+X*X') - Z'*Z, 1) / norm (E,1)

fprintf ('CHOLMOD lchol(sparse(A)) speedup over chol(sparse(A)): %6.1f\n', ...
    s1 / t1) ;

fprintf ('CHOLMOD sparse update speedup vs MATLAB DENSE update:  %6.1f\n', ...
    s3 / t3) ;

clear E S L R LD X C D Z
clear err s1 s2 s3 t1 t2 t3 n
