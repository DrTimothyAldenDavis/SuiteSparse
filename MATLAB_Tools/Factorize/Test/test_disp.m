function test_disp
%TEST_DISP test the display method of the factorize object
%
% Example
%   test_disp
%
% See also factorize, factorize1, test_all.

% Copyright 2009, Timothy A. Davis, University of Florida

tol = 1e-12 ;

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
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.Q) || ~isempty (F.R) || ~isempty (F.q))
    error ('invalid contents') ;
end
if (F.is_inverse || ~isa (F, 'factorization_dense_lu'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_dense_lu'))
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
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.Q) || ~isempty (F.R))
    error ('invalid contents') ;
end
if (F.is_inverse || ~isa (F, 'factorization_sparse_lu'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_sparse_lu'))
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
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.Q) || ~isempty (F.L) || ~isempty (F.U) || ~isempty (F.q) ...
 || ~isempty (F.p))
    error ('invalid contents') ;
end
if (F.is_inverse || ~isa (F, 'factorization_dense_chol'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_dense_chol'))
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
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.Q) || ~isempty (F.R) || ~isempty (F.U) || ~isempty (F.p))
    error ('invalid contents') ;
end
if (F.is_inverse || ~isa (F, 'factorization_sparse_chol'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (F, 'factorization_sparse_chol'))
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
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.L) || ~isempty (F.U) || ~isempty (F.p) || ~isempty (F.q))
    error ('invalid contents') ;
end
if (F.is_inverse || ~isa (F, 'factorization_dense_qr'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_dense_qr'))
    error ('invalid contents') ;
end

fprintf ('Dense QR factorization of A'':\n') ;
F = factorize (A') ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

Q = F.Q ;
R = F.R ;
C = F.A ;
err = norm (C' - Q*R, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.L) || ~isempty (F.U) || ~isempty (F.p) || ~isempty (F.q))
    error ('invalid contents') ;
end
if (F.is_inverse || ~isa (F, 'factorization_dense_qrt'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_dense_qrt'))
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
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.L) || ~isempty (F.U) || ~isempty (F.p))
    error ('invalid contents') ;
end
if (F.is_inverse || ~isa (F, 'factorization_sparse_qr'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_sparse_qr'))
    error ('invalid contents') ;
end

fprintf ('Sparse QR factorization of A'':\n') ;
F = factorize (A') ;
disp (F) ;
S = inverse (F) ;
disp (S) ;

Q = F.Q ;
R = F.R ;
p = F.p ;
C = F.A ;
err = norm ((p*C)*(p*C)' - R'*R, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (~isempty (F.L) || ~isempty (F.U) || ~isempty (F.q))
    error ('invalid contents') ;
end
if (F.is_inverse || ~isa (F, 'factorization_sparse_qrt'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_sparse_qrt'))
    error ('invalid contents') ;
end

fprintf ('\nAll disp tests passed\n') ;

