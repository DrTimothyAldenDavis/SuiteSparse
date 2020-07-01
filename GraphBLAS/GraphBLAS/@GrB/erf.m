function C = erf (G)
%ERF error function.
% C = erf (G) computes the error function of each entry of G.
% G must be real.
%
% See also GrB/erfc, erfcx, erfinv, erfcinv.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;
if (contains (type, 'complex'))
    error ('input must be real') ;
end
if (~gb_isfloat (type))
    op = 'erf.double' ;
else
    op = 'erf' ;
end

C = GrB (gbapply (op, G)) ;

