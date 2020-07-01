function C = gb_to_real_if_imag_zero (G)
%GB_TO_REAL_IF_IMAG_ZERO convert complex matrix to real if imag(G) is zero

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (contains (gbtype (G), 'complex') && ...
    gbnvals (gbselect ('nonzero', gbapply ('cimag', G))) == 0)
    C = gbapply ('creal', G) ;
else
    C = G ;
end

