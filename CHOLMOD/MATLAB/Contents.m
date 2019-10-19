% CHOLMOD: a sparse supernodal Cholesky update/downdate package
%
% cholmod        x = cholmod (A,b), computes x=A\b when A is symmetric and
%                positive definite, just faster
%
% chol2          same as MATLAB chol(sparse(A)), just faster
% lchol          L = lchol (A) computes an LL' factorization
% ldlchol        LD = ldlchol (A) computes an LDL' factorization
%
% ldlupdate      LD = ldlupdate (LD,C,...) updates an LDL' factorization
% resymbol       L = resymbol (L,A) recomputes symbolic LL or LDL' factorization
% ldlsolve       x = ldlsolve (LD,b) solves Ax=b using an LDL' factorization
% ldlsplit       [L,D] = ldlsplit (LD) splits LD into L and D.
%
% metis          interface to METIS node-nested-dissection
% nesdis         interface to CHOLMOD's nested-dissection (based on METIS)
% septree        prune a separator tree
% bisect         interface to METIS' node bisector
% analyze        order and analyze using CHOLMOD
%
% etree2         same as MATLAB "etree", just faster and more reliable
% sparse2        same as MATLAB "sparse", just faster
% symbfact2      same as MATLAB "symbfact", just faster and more reliable
%
% sdmult         same as MATLAB S*F or S'*F (S sparse, F full), just faster
%
% mread          read a sparse matrix in Matrix Market format
% mwrite         write a sparse matrix in Matrix Market format
% spsym          determine the symmetry of a sparse matrix
%
% ldl_normest    err = ldl_normest (A,L,D), error in LDL' factorization
% lu_normest     err = lu_normest (A,L,U), error in LU factorization
% cholmod_demo   test CHOLMOD with random sparse matrices
% Test           directory for testing CHOLMOD with UF sparse matrix collection

%   Copyright 2006, Timothy A. Davis
%   http://www.cise.ufl.edu/research/sparse
