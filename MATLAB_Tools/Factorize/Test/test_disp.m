function test_disp
%TEST_DISP test the display method of the factorize object
%
% Example
%   test_disp
%
% See also factorize, factorize1, test_all.

% Copyright 2009, Timothy A. Davis, University of Florida

fprintf ('Dense LU factorization:\n') ;
A = rand (3) ;
F = factorize1 (A) ;
disp (F) ;
S = inverse (F) ;
disp (S) ;
F = factorize (A) ; 
disp (F) ;
S = inverse (F) ;
disp (S) ;

L = F.L ;
U = F.U ;
p = F.p ;
C = F.A ;
err = norm (p*C - L*U, 1) ;
if (err > 1e-14)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.Q) || ~isempty (F.R) || ~isempty (F.q))
    error ('invalid contents') ;
end
if (F.is_inverse || F.kind ~= 8)
    error ('invalid contents') ;
end
if (~(S.is_inverse) || S.kind ~= 8)
    error ('invalid contents') ;
end

fprintf ('Sparse LU factorization:\n') ;
A = sparse (A) ;
F = factorize1 (A) ;
disp (F) ;
S = inverse (F) ;
disp (S) ;
F = factorize (A) ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

L = F.L ;
U = F.U ;
p = F.p ;
q = F.q ;
C = F.A ;
err = norm (p*C*q - L*U, 1) ;
if (err > 1e-14)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.Q) || ~isempty (F.R))
    error ('invalid contents') ;
end
if (F.is_inverse || F.kind ~= 7)
    error ('invalid contents') ;
end
if (~(S.is_inverse) || S.kind ~= 7)
    error ('invalid contents') ;
end

fprintf ('Dense Cholesky factorization:\n') ;
A = A*A' + eye (3) ;
F = factorize (A) ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

R = F.R ;
C = F.A ;
err = norm (C - R'*R, 1) ;
if (err > 1e-14)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.Q) || ~isempty (F.L) || ~isempty (F.U) || ~isempty (F.q) ...
 || ~isempty (F.p))
    error ('invalid contents') ;
end
if (F.is_inverse || F.kind ~= 6)
    error ('invalid contents') ;
end
if (~(S.is_inverse) || S.kind ~= 6)
    error ('invalid contents') ;
end

fprintf ('Sparse Cholesky factorization:\n') ;
A = sparse (A) ;
F = factorize (A) ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

L = F.L ;
q = F.q ;
C = F.A ;
err = norm (q*C*q' - L*L', 1) ;
if (err > 1e-14)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.Q) || ~isempty (F.R) || ~isempty (F.U) || ~isempty (F.p))
    error ('invalid contents') ;
end
if (F.is_inverse || F.kind ~= 5)
    error ('invalid contents') ;
end
if (~(S.is_inverse) || S.kind ~= 5)
    error ('invalid contents') ;
end

fprintf ('Dense QR factorization:\n') ;
A = rand (3,2) ;
F = factorize (A) ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

Q = F.Q ;
R = F.R ;
C = F.A ;
err = norm (C - Q*R, 1) ;
if (err > 1e-14)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.L) || ~isempty (F.U) || ~isempty (F.p) || ~isempty (F.q))
    error ('invalid contents') ;
end
if (F.is_inverse || F.kind ~= 2)
    error ('invalid contents') ;
end
if (~(S.is_inverse) || S.kind ~= 2)
    error ('invalid contents') ;
end

F = factorize (A') ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

Q = F.Q ;
R = F.R ;
C = F.A ;
err = norm (C' - Q*R, 1) ;
if (err > 1e-14)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.L) || ~isempty (F.U) || ~isempty (F.p) || ~isempty (F.q))
    error ('invalid contents') ;
end
if (F.is_inverse || F.kind ~= 4)
    error ('invalid contents') ;
end
if (~(S.is_inverse) || S.kind ~= 4)
    error ('invalid contents') ;
end

fprintf ('Sparse QR factorization:\n') ;
A = sparse (A) ;
F = factorize (A) ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

Q = F.Q ;
R = F.R ;
q = F.q ;
C = F.A ;
err = norm ((C*q)'*(C*q) - R'*R, 1) ;
if (err > 1e-14)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.L) || ~isempty (F.U) || ~isempty (F.p))
    error ('invalid contents') ;
end
if (F.is_inverse || F.kind ~= 1)
    error ('invalid contents') ;
end
if (~(S.is_inverse) || S.kind ~= 1)
    error ('invalid contents') ;
end

F = factorize (A') ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

Q = F.Q ;
R = F.R ;
p = F.p ;
C = F.A ;
err = norm ((p*C)*(p*C)' - R'*R, 1) ;
if (err > 1e-14)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.L) || ~isempty (F.U) || ~isempty (F.q))
    error ('invalid contents') ;
end
if (F.is_inverse || F.kind ~= 3)
    error ('invalid contents') ;
end
if (~(S.is_inverse) || S.kind ~= 3)
    error ('invalid contents') ;
end

fprintf ('\nAll disp tests passed\n') ;

