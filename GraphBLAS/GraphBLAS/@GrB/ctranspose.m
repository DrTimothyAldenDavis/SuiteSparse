function C = ctranspose (G)
%CTRANSPOSE C = G', transpose a GraphBLAS matrix.
% C = G' is the complex conjugate transpose of G.
%
% See also GrB.trans, GrB/transpose, GrB/conj.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (contains (gbtype (G), 'complex'))
    desc.in0 = 'transpose' ;
    C = GrB (gbapply ('conj', G, desc)) ;
else
    C = GrB (gbtrans (G)) ;
end

