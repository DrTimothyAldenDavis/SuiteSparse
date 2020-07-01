function C = plus (A, B)
%PLUS sparse matrix addition, C = A+B.
% C = A+B adds the two matrices A and B.  If A and B are matrices, the
% pattern of C is the set union of A and B.  If one of A or B is a
% nonzero scalar, the scalar is expanded into a dense matrix the size of
% the other matrix, and the result is a dense matrix.
%
% See also GrB.eadd, GrB/minus, GrB/uminus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

C = GrB (gb_eadd (A, '+', B)) ;

