function C = erfc (G)
%ERFC complementary error function.
% C = erfc (G) is the complementary error function of each entry of G.
% Since erfc (0) = 1, the result is a full matrix.  G must be real.
%
% See also GrB/erf, erfcx, erfinv, erfcinv.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;
if (contains (type, 'complex'))
    error ('input must be real') ;
end
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('erfc', gbfull (G, type))) ;

