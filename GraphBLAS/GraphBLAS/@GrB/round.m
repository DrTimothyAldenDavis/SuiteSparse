function C = round (G)
%ROUND round entries of a matrix to the nearest integers.
% C = round (G) rounds the entries of G to the nearest integers.
%
% Note: the additional parameters of the built-in MATLAB round function,
% round(x,n) and round (x,n,type), are not supported.
%
% See also GrB/ceil, GrB/floor, GrB/fix.

% FUTURE: round (x,n) and round (x,n,type)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

Q = G.opaque ;

if (gb_isfloat (gbtype (Q)) && gbnvals (Q) > 0)
    C = GrB (gbapply ('round', Q)) ;
else
    C = G ;
end

