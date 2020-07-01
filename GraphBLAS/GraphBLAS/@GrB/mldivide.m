function C = mldivide (A, B)
% C = A\B, matrix left division.
% If A is a scalar, then C = A.\B is computed; see 'help ldivide'.
% Otherwise, C is computed by first converting A and B to MATLAB sparse
% matrices, and then C=A\B is computed using the MATLAB backslash.
%
% See also GrB/mrdivide.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isscalar (A))
    C = rdivide (B, A) ;
else
    C = GrB (builtin ('mldivide', double (A), double (B))) ;
end

