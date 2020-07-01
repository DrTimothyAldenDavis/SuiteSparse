function C = double (G)
%DOUBLE cast a GraphBLAS sparse matrix to a MATLAB sparse double matrix.
% C = double (G) typecasts the GraphBLAS matrix G into a MATLAB sparse
% double matrix C, either real or complex.  Explicit zeros are dropped.
%
% To typecast the matrix G to a GraphBLAS sparse double (real) matrix
% instead, use C = GrB (G, 'double').  Explicit zeros are kept in C.
%
% See also GrB/cast, GrB, GrB/complex, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (contains (gbtype (G), 'complex'))
    C = gbsparse (G, 'double complex') ;
else
    C = gbsparse (G, 'double') ;
end

