% Demo for colamd:  column approximate minimum degree ordering algorithm.
% 
% The following m-files and mexFunctions provide alternative sparse matrix
% ordering methods for MATLAB.  They are typically faster (sometimes much
% faster) and typically provide better orderings than their MATLAB counterparts:
% 
%	colamd		a replacement for colmmd.
%
%			Typical usage:  p = colamd (A) ;
%
%	symamd		a replacement for symmmd.  Based on colamd.
%
%			Typical usage:  p = symamd (A) ;
%
% For a description of the methods used, see the colamd.c file.
%
% COLAMD Version 2.5.
% http://www.cise.ufl.edu/research/sparse/colamd/
%

% Minor changes:  in MATLAB 7, symmmd and colmmd are flagged as "obsolete".
% This demo checks if they exist, so it should still work when they are removed.

%-------------------------------------------------------------------------------
% Print the introduction, the help info, and compile the mexFunctions
%-------------------------------------------------------------------------------

fprintf (1, '\n-----------------------------------------------------------\n') ;
fprintf (1, 'Colamd/symamd demo.') ;
fprintf (1, '\n-----------------------------------------------------------\n') ;
help colamd_demo ;

fprintf (1, '\n-----------------------------------------------------------\n') ;
fprintf (1, 'Colamd help information:') ;
fprintf (1, '\n-----------------------------------------------------------\n') ;
help colamd ;

fprintf (1, '\n-----------------------------------------------------------\n') ;
fprintf (1, 'Symamd help information:') ;
fprintf (1, '\n-----------------------------------------------------------\n') ;
help symamd ;

%-------------------------------------------------------------------------------
% Solving Ax=b
%-------------------------------------------------------------------------------

n = 100 ;
fprintf (1, '\n-----------------------------------------------------------\n') ;
fprintf (1, 'Solving Ax=b for a small %d-by-%d random matrix:', n, n) ;
fprintf (1, '\n-----------------------------------------------------------\n') ;
fprintf (1, '\nNOTE: Random sparse matrices are AWFUL test cases.\n') ;
fprintf (1, 'They''re just easy to generate in a demo.\n') ;

% set up the system

rand ('state', 0) ;
randn ('state', 0) ;
spparms ('default') ;
A = sprandn (n, n, 5/n) + speye (n) ;
b = (1:n)' ;

fprintf (1, '\n\nSolving via lu (PAQ = LU), where Q is from colamd:\n') ;
q = colamd (A) ;
I = speye (n) ;
Q = I (:, q) ;
[L,U,P] = lu (A*Q) ;
fl = luflops (L, U) ;
x = Q * (U \ (L \ (P * b))) ;
fprintf (1, '\nFlop count for [L,U,P] = lu (A*Q):          %d\n', fl) ;
fprintf (1, 'residual:                                     %e\n', norm (A*x-b));

try
fprintf (1, '\n\nSolving via lu (PAQ = LU), where Q is from colmmd:\n') ;
q = colmmd (A) ;
I = speye (n) ;
Q = I (:, q) ;
[L,U,P] = lu (A*Q) ;
fl = luflops (L, U) ;
x = Q * (U \ (L \ (P * b))) ;
fprintf (1, '\nFlop count for [L,U,P] = lu (A*Q):          %d\n', fl) ;
fprintf (1, 'residual:                                     %e\n', norm (A*x-b));
catch
fprintf (1, 'colmmd is obsolete\n') ;
end

fprintf (1, '\n\nSolving via lu (PA = LU), without regard for sparsity:\n') ;
[L,U,P] = lu (A) ;
fl = luflops (L, U) ;
x = U \ (L \ (P * b)) ;
fprintf (1, '\nFlop count for [L,U,P] = lu (A*Q):          %d\n', fl) ;
fprintf (1, 'residual:                                     %e\n', norm (A*x-b));

%-------------------------------------------------------------------------------
% Large demo for colamd
%-------------------------------------------------------------------------------

fprintf (1, '\n-----------------------------------------------------------\n') ;
fprintf (1, 'Large demo for colamd (symbolic analysis only):') ;
fprintf (1, '\n-----------------------------------------------------------\n') ;

rand ('state', 0) ;
randn ('state', 0) ;
spparms ('default') ;
n = 1000 ;
fprintf (1, 'Generating a random %d-by-%d sparse matrix.\n', n, n) ;
A = sprandn (n, n, 5/n) + speye (n) ;

fprintf (1, '\n\nUnordered matrix:\n') ;
lnz = symbfact (A, 'col') ;
fprintf (1, 'nz in Cholesky factors of A''A:            %d\n', sum (lnz)) ;
fprintf (1, 'flop count for Cholesky of A''A:           %d\n', sum (lnz.^2)) ;

tic ;
p = colamd (A) ;
t = toc ;
lnz = symbfact (A (:,p), 'col') ;
fprintf (1, '\n\nColamd run time:                          %f\n', t) ;
fprintf (1, 'colamd ordering quality: \n') ;
fprintf (1, 'nz in Cholesky factors of A(:,p)''A(:,p):  %d\n', sum (lnz)) ;
fprintf (1, 'flop count for Cholesky of A(:,p)''A(:,p): %d\n', sum (lnz.^2)) ;

try
tic ;
p = colmmd (A) ;
t = toc ;
lnz = symbfact (A (:,p), 'col') ;
fprintf (1, '\n\nColmmd run time:                          %f\n', t) ;
fprintf (1, 'colmmd ordering quality: \n') ;
fprintf (1, 'nz in Cholesky factors of A(:,p)''A(:,p):  %d\n', sum (lnz)) ;
fprintf (1, 'flop count for Cholesky of A(:,p)''A(:,p): %d\n', sum (lnz.^2)) ;
catch
fprintf (1, 'colmmd is obsolete\n') ;
end

%-------------------------------------------------------------------------------
% Large demo for symamd
%-------------------------------------------------------------------------------

fprintf (1, '\n-----------------------------------------------------------\n') ;
fprintf (1, 'Large demo for symamd (symbolic analysis only):') ;
fprintf (1, '\n-----------------------------------------------------------\n') ;

fprintf (1, 'Generating a random symmetric %d-by-%d sparse matrix.\n', n, n) ;
A = A+A' ;

fprintf (1, '\n\nUnordered matrix:\n') ;
lnz = symbfact (A, 'sym') ;
fprintf (1, 'nz in Cholesky factors of A:       %d\n', sum (lnz)) ;
fprintf (1, 'flop count for Cholesky of A:      %d\n', sum (lnz.^2)) ;

tic ;
p = symamd (A) ;
t = toc ;
lnz = symbfact (A (p,p), 'sym') ;
fprintf (1, '\n\nSymamd run time:                   %f\n', t) ;
fprintf (1, 'symamd ordering quality: \n') ;
fprintf (1, 'nz in Cholesky factors of A(p,p):  %d\n', sum (lnz)) ;
fprintf (1, 'flop count for Cholesky of A(p,p): %d\n', sum (lnz.^2)) ;

try
tic ;
p = symmmd (A) ;
t = toc ;
lnz = symbfact (A (p,p), 'sym') ;
fprintf (1, '\n\nSymmmd run time:                   %f\n', t) ;
fprintf (1, 'symmmd ordering quality: \n') ;
fprintf (1, 'nz in Cholesky factors of A(p,p):  %d\n', sum (lnz)) ;
fprintf (1, 'flop count for Cholesky of A(p,p): %d\n', sum (lnz.^2)) ;
catch
fprintf (1, 'symmmd is obsolete\n') ;
end
