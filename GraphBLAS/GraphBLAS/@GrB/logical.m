function C = logical (G)
%LOGICAL typecast a GraphBLAS matrix to MATLAB sparse logical matrix.
% C = logical (G) typecasts the GraphBLAS matrix G to into a MATLAB sparse
% logical matrix.
%
% To typecast the matrix G to a GraphBLAS sparse logical matrix instead,
% use C = GrB (G, 'logical').
%
% See also cast, GrB, GrB/double, GrB/complex, GrB/single, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
C = gbsparse (G, 'logical') ;

