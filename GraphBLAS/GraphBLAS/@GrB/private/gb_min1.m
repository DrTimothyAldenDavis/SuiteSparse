function C = gb_min1 (op, A)
%GB_MIN1 single-input min
% Implements C = min (A)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[m, n] = gbsize (A) ;
if (m == 1 || n == 1)
    % C = min (A) for a vector A results in a scalar C
    C = gb_minall (op, A) ;
else
    C = gb_minbycol (op, A) ;
end

