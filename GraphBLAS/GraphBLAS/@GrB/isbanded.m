function s = isbanded (A, lo, hi)
%ISBANDED true if A is a banded matrix.
% isbanded (A, lo, hi) is true if the bandwidth of A is between lo and hi.
%
% See also GrB/istril, GrB/istriu, GrB/bandwidth.

% FUTURE: this will be much faster when 'gb_bandwidth' is a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved.  http://suitesparse.com   See GraphBLAS/Doc/License.txt.

if (isobject (A))
    A = A.opaque ;
end

lo = gb_get_scalar (lo) ;
hi = gb_get_scalar (hi) ;

[alo, ahi] = gb_bandwidth (A) ;
s = (alo <= lo) & (ahi <= hi) ;

