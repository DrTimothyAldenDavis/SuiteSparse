function C = gb_scalar_to_full (m, n, type, scalar)
%GB_DENSE expand a scalar into a dense matrix

C = gbsubassign (gbnew (m, n, type), gbfull (scalar)) ;
