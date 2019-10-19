function [L,U,Off,Pnum,Q,Rs] = kluaf (A)
%
% [L,U,Off,Pnum,Q,Rs] kluaf (A)
%
% The factorization is L*U + Off = Rs \ (A (Pnum,Q))

symtol = 1e-3 ;

[P,Q,R,Lnz,Info1] = klua (A) ;
if (isreal (A))
    [L,U,Off,Pnum,Rs,Info2] = kluf (A, P,Q,R,Lnz,Info1) ;
else
    [L,U,Off,Pnum,Rs,Info2] = klufz (A, P,Q,R,Lnz,Info1) ;
end

if (nargout == 0)
    % L
    % U
    % Off
    % Pnum
    % Q
    % Rs
    % return LU norm
    L = lu_normest ( -Off + (Rs \ (A (Pnum,Q))), L, U) ;
end

% an UMFPACK v4.3 mexFunction:
% klu_flop = luflop (L,U) ;
