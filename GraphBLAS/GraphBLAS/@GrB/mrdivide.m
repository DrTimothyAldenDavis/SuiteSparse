function C = mrdivide (A, B)
% C = A/B, matrix right division.
% If A is a scalar, then C = A./B is computed; see 'help rdivide'.
% Otherwise, C is computed by first converting A and B to MATLAB sparse
% matrices, and then C=A/B is computed using the MATLAB backslash.
%
% See also GrB/mldivide.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isscalar (B))
    C = rdivide (A, B) ;
else
    C = GrB (builtin ('mrdivide', double (A), double (B))) ;
end

