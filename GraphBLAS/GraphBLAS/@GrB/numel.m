function s = numel (G)
%NUMEL the maximum number of entries in a matrix.
% numel (G) is m*n for the m-by-n GraphBLAS matrix G.  If m, n, or m*n
% exceed flintmax (2^53), the result is returned as a vpa symbolic value,
% to avoid integer overflow.
%
% See also GrB/nnz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
s = gb_numel (G) ;

