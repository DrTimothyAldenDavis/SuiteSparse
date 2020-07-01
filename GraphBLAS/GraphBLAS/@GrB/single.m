function C = single (G)
%SINGLE cast a GraphBLAS matrix to MATLAB full single matrix.
% C = single (G) typecasts the GrB matrix G to a MATLAB full single
% matrix.  The result C is full since MATLAB does not support sparse
% single matrices.  C is real if G is real, and complex if G is complex.
%
% To typecast the matrix G to a GraphBLAS sparse single matrix instead,
% use C = GrB (G, 'single').  To typecast to a sparse single complex
% matrix, use G = GrB (G, 'single complex').
%
% See also GrB, GrB/double, GrB/complex, GrB/logical, GrB/int8, GrB/int16,
% GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32, GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
desc.kind = 'full' ;
if (contains (gbtype (G), 'complex'))
    C = gbfull (G, 'single complex', complex (single (0)), desc) ;
else
    C = gbfull (G, 'single', single (0), desc) ;
end

