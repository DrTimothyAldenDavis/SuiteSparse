function SuiteSparse_paths
%SuiteSparse_paths: adds paths to all SuiteSparse mexFunctions
% A Suite of Sparse matrix packages, authored or co-authored by Tim Davis.
%
% Packages in SuiteSparse:
%
% GraphBLAS      graph algorithms via sparse linear algebra (graphblas.org)
% Mongoose       graph partitioner
% SPEX           solve sparse Ax=b exactly
% UMFPACK        sparse LU factorization (multifrontal)
% CHOLMOD        sparse Cholesky factorization, and many other operations
% AMD            sparse symmetric approximate minimum degree ordering
% COLAMD         sparse column approximate minimum degree ordering
% CAMD           constrained AMD
% CCOLAMD        constrained COLAMD
% CSparse        a Concise Sparse matrix package (32-bit or 64-bit, real only)
% CXSparse       extended version of CSparse (32-bit/64-bit/real/complex)
% ssget          interface to SuiteSparse Matrix Collection
% KLU            sparse LU factorization (left-looking)
% BTF            permutation to block triangular form (like dmperm)
% LDL            sparse LDL' factorization
% SuiteSparseCollection   tools for managing the SuiteSparse Matrix Collection
% RBio           read/write Rutherford/Boeing files
% SSMULT         sparse matrix times sparse matrix
% MESHND         2D and 3D regular mesh generation and nested dissection
% FACTORIZE      an object-oriented solver for x=A\b
% SPARSEINV      sparse inverse subset; computes entries of inv(sparse(A))
% MATLAB_Tools   various simple m-files and demos
% SuiteSparseQR  sparse QR factorization
% spqr_rank      MATLAB toolbox for sparse rank deficient matrices
%
% Example:
%    SuiteSparse_paths  % adds all paths to the SuiteSparse mexFunctions
%
% This method adds the mexFunction paths to all SuiteSparse mexFunctions, and
% can be used at the start of a MATLAB session.  The mexFunctions must also be
% installed via SuiteSparse_install.
%
% You must run this m-file while in the SuiteSparse folder containing this
% m-file.
%
% Copyright (c) 1990-2022, Timothy A. Davis, http://suitesparse.com.
% See each package for its license.

paths = { } ;
SuiteSparse = pwd ;

paths = add_to_path (paths, SuiteSparse) ;
paths = add_to_path (paths, [SuiteSparse '/UMFPACK/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/CHOLMOD/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/AMD/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/COLAMD/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/CCOLAMD/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/CAMD/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/ssget']) ;
paths = add_to_path (paths, [SuiteSparse '/CXSparse/MATLAB/Demo']) ;
paths = add_to_path (paths, [SuiteSparse '/CXSparse/MATLAB/CSparse']) ;
paths = add_to_path (paths, [SuiteSparse '/LDL/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/BTF/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/KLU/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/SPQR/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/RBio/RBio']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/Factorize']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/MESHND']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/LINFACTOR']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/find_components']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/GEE']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/shellgui']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/waitmex']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/spqr_rank']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/spqr_rank/SJget']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/SuiteSparseCollection']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/SSMULT']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/dimacs10']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/spok']) ;
paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/sparseinv']) ;
paths = add_to_path (paths, [SuiteSparse '/Mongoose/MATLAB']) ;
paths = add_to_path (paths, [SuiteSparse '/GraphBLAS/GraphBLAS/build']) ;
paths = add_to_path (paths, [SuiteSparse '/GraphBLAS/GraphBLAS/demo']) ;
paths = add_to_path (paths, [SuiteSparse '/GraphBLAS/GraphBLAS']) ;
paths = add_to_path (paths, [SuiteSparse '/SPEX/SPEX_Left_LU/MATLAB']) ;

cd (SuiteSparse)

fprintf ('\nSuiteSparse installed for MATLAB\n') ;

%-------------------------------------------------------------------------------
function paths = add_to_path (paths, newpath)
% add a path
cd (newpath) ;
addpath (newpath) ;
paths = [paths { newpath } ] ;
