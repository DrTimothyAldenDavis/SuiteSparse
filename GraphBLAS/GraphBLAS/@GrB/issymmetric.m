function s = issymmetric (G, option)
%ISSYMMETRIC Determine if a GraphBLAS matrix is real or complex symmetric.
% issymmetric (G) is true if G equals G.' and false otherwise.
% issymmetric (G, 'skew') is true if G equals -G.' and false otherwise.
% issymmetric (G, 'nonskew') is the same as issymmetric (G).
%
% See also GrB/ishermitian.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (nargin < 2)
    option = 'nonskew' ;
end

s = gb_issymmetric (G, option, false) ;

