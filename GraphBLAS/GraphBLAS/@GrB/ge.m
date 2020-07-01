function C = ge (A, B)
%A >= B greater than or equal to.
% C = (A >= B) is an element-by-element comparison of A and B.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/lt, GrB/le, GrB/gt, GrB/ne, GrB/eq.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = le (B, A) ;

