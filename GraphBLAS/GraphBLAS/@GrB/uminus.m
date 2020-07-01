function C = uminus (G)
%UMINUS negate a matrix.
% C = -G negates the entries of the matrix G.
%
% See also GrB.apply, GrB/minus, GrB/uplus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
C = GrB (gbapply ('-', G)) ;

