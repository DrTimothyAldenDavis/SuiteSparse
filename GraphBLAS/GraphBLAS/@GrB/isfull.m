function s = isfull (A)
%GRB.ISFULL determine if all entries are present.
% For a GraphBLAS matrix, or a MATLAB sparse matrix, GrB.isfull (A) is true
% if numel (A) == nnz (A).  A can be a GraphBLAS matrix, or a MATLAB sparse
% or full matrix.  GrB.isfull (A) is always true if A is a MATLAB full
% matrix.
%
% See also GrB/issparse, GrB/full.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (A))
    % GraphBLAS matrix
    A = A.opaque ;
    s = gb_isfull (A) ;
elseif (issparse (A))
    % MATLAB sparse matrix
    s = (numel (A) == nnz (A)) ;
else
    % MATLAB full matrix, string, struct, etc
    s = true ;
end

