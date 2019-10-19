% Welcome to The SuiteSparse : a Suite of Sparse matrix packages, containing a
% collection of sparse matrix packages authored or co-authored by Tim Davis.
%
%   SuiteSparse_install - compiles and installs all of the SuiteSparse, and runs
%               several demos and tests.
%
%-------------------
% Ordering methods:
%-------------------
%
%   amd          - approximate minimum degree ordering.
%   colamd       - column approximate minimum degree ordering.
%   symamd       - symmetrix approximate min degree ordering based on colamd.
%   camd         - constrained amd.
%   ccolamd      - constrained colamd.
%   csymamd      - constrained symamd.
%
%---------------------------------------------------------------
% CHOLMOD: a sparse supernodal Cholesky update/downdate package:
%---------------------------------------------------------------
%
%   cholmod      - computes x=A\b when A is symmetric and positive definite.
%   chol2        - same as MATLAB chol(sparse(A)), just faster.
%   lchol        - computes an LL' factorization.
%   ldlchol      - computes an LDL' factorization.
%   ldlupdate    - updates an LDL' factorization.
%   resymbol     - recomputes symbolic LL or LDL' factorization.
%   ldlsolve     - solves Ax=b using an LDL' factorization.
%   ldlsplit     - splits LD into L and D.
%   metis        - interface to METIS node-nested-dissection.
%   nesdis       - interface to CHOLMOD's nested-dissection (based on METIS).
%   septree      - prune a separator tree.
%   bisect       - interface to METIS' node bisector.
%   analyze      - order and analyze using CHOLMOD.
%   etree2       - same as MATLAB "etree", just faster and more reliable.
%   sparse2      - same as MATLAB "sparse", just faster.
%   symbfact2    - same as MATLAB "symbfact", just faster and more reliable.
%   sdmult       - same as MATLAB S*F or S'*F (S sparse, F full), just faster.
%   ldl_normest  - compute error in LDL' factorization.
%   lu_normest   - compute error in LU factorization.
%
%------------------------------------------
% CSPARSE: a Concise Sparse matrix package:
%------------------------------------------
%
%   Matrices used in CSparse must in general be either sparse and real,
%   or dense vectors.  Ordering methods can accept any sparse matrix.
%
%   cs_add       - sparse matrix addition.
%   cs_amd       - approximate minimum degree ordering.
%   cs_chol      - sparse Cholesky factorization.
%   cs_cholsol   - solve A*x=b using a sparse Cholesky factorization.
%   cs_counts    - column counts for sparse Cholesky factor L.
%   cs_dmperm    - maximum matching or Dulmage-Mendelsohn permutation.
%   cs_dmsol     - x=A\b using the coarse Dulmage-Mendelsohn decomposition.
%   cs_dmspy     - plot the Dulmage-Mendelsohn decomposition of a matrix.
%   cs_droptol   - remove small entries from a sparse matrix.
%   cs_esep      - find an edge separator of a symmetric matrix A
%   cs_etree     - elimination tree of A or A'*A.
%   cs_gaxpy     - sparse matrix times vector.
%   cs_lsolve    - solve a sparse lower triangular system L*x=b.
%   cs_ltsolve   - solve a sparse upper triangular system L'*x=b.
%   cs_lu        - sparse LU factorization, with fill-reducing ordering.
%   cs_lusol     - solve Ax=b using LU factorization.
%   cs_make      - compiles CSparse for use in MATLAB.
%   cs_multiply  - sparse matrix multiply.
%   cs_nd        - generalized nested dissection ordering.
%   cs_nsep      - find a node separator of a symmetric matrix A.
%   cs_permute   - permute a sparse matrix.
%   cs_print     - print the contents of a sparse matrix.
%   cs_qr        - sparse QR factorization.
%   cs_qleft     - apply Householder vectors on the left.
%   cs_qright    - apply Householder vectors on the right.
%   cs_qrsol     - solve a sparse least-squares problem.
%   cs_randperm  - random permutation.
%   cs_sep       - convert an edge separator into a node separator.
%   cs_scc       - strongly-connected components of a square sparse matrix.
%   cs_sparse    - convert a triplet form into a sparse matrix.
%   cs_sqr       - symbolic sparse QR factorization.
%   cs_symperm   - symmetric permutation of a symmetric matrix.
%   cs_transpose - transpose a real sparse matrix.
%   cs_updown    - rank-1 update/downdate of a sparse Cholesky factorization.
%   cs_usolve    - solve a sparse upper triangular system U*x=b.
%   cs_utsolve   - solve a sparse lower triangular system U'*x=b.
%   cspy         - plot a sparse matrix in color.
%
%-------------------------------
% LDL: Sparse LDL factorization:
%-------------------------------
% 
%   ldlsparse   - LDL' factorization of a real, sparse, symmetric matrix.
%   ldlrow      - an m-file description of the algorithm used by LDL.
%
%-----------------------------------------------
% UMFPACK: the Unsymmetric MultiFrontal Package:
%-----------------------------------------------
%
%   umfpack           - computes x=A\b, x=A/b, or lu (A) for a sparse matrix A
%   umfpack_details   - details on all the options for using umfpack in MATLAB
%   umfpack_report    - prints optional control settings and statistics
%   umfpack_btf       - factorize A using a block triangular form
%   umfpack_solve     - x = A\b or x = b/A
%   lu_normest        - estimates norm (L*U-A,1) without forming L*U-A
%                       (duplicate of CHOLMOD/lu_normest, for completeness)
%   luflop            - given L and U, computes # of flops required
%
%-------------------------------------------------------------------------------
%
% For help on compiling the SuiteSparse, demos, testing functions, etc., please
% see the help for each individual package.
%
% Copyright 2006, Timothy A. Davis
% http://www.cise.ufl.edu/research/sparse

help SuiteSparse
