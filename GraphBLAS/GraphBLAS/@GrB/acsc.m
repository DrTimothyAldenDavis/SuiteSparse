function C = acsc (G)
%ACSC inverse cosecant.
% C = acsc (G) is the inverse cosecant of each entry of G.
% Since acsc (0) is nonzero, C is a full matrix.
%
% See also GrB/csc, GrB/csch, GrB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;

if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gb_trig ('asin', gbapply ('minv', gbfull (G, type)))) ;

