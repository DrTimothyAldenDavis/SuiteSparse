function test_disp
%TEST_DISP test the display method of the factorize object
%
% Example
%   test_disp
%
% See also factorize, test_all.

% Copyright 2011, Timothy A. Davis, University of Florida.

reset_rand ;
tol = 1e-12 ;

%-------------------------------------------------------------------------------
% dense LU
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense LU factorization:\n') ;
A = rand (3) ;

F = factorize (A, [ ], 1) ; 
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
L = f.L ;
U = f.U ;
P = f.P ;
C = F.A ;
err = norm (P*C - L*U, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_lu_dense'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_lu_dense'))
    error ('invalid contents') ;
end

fprintf ('\nDense LU With an imaginary F.alpha:\n') ;
alpha = (pi + 2i) ;
F = alpha*F ;
display (F) ;
b = rand (3,1) ;
x = F\b ;
y = (alpha*A)\b ;
err = norm (x-y) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end

%-------------------------------------------------------------------------------
% sparse LU
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse LU factorization:\n') ;
A = sparse (A) ;

F = factorize (A, [ ], 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
L = f.L ;
U = f.U ;
P = f.P ;
Q = f.Q ;
C = F.A ;
err = norm (P*C*Q - L*U, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_lu_sparse'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_lu_sparse'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% dense Cholesky
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense Cholesky factorization:\n') ;
A = A*A' + eye (3) ;
F = factorize (A, [ ], 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
R = f.R ;
C = F.A ;
err = norm (C - R'*R, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_chol_dense'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_chol_dense'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% sparse Cholesky
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse Cholesky factorization:\n') ;
A = sparse (A) ;
F = factorize (A, [ ], 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
L = f.L ;
P = f.P ;
C = F.A ;
err = norm (P'*C*P - L*L', 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_chol_sparse'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (F, 'factorization_chol_sparse'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% dense QR of A
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense QR factorization:\n') ;
A = rand (3,2) ;
F = factorize (A, [ ], 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
Q = f.Q ;
R = f.R ;
C = F.A ;
err = norm (C - Q*R, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_qr_dense'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_qr_dense'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% dense COD of A
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense COD factorization:\n') ;
F = factorize (A, 'cod', 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
U = f.U ;
R = f.R ;
V = f.V ;
C = F.A ;
err = norm (C - U*R*V', 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_cod_dense'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_cod_dense'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% sparse COD of A
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse COD factorization:\n') ;
F = factorize (sparse (A), 'cod', 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
U = f.U ;
R = f.R ;
V = f.V ;
C = F.A ;
display (U)
display (V)
U = cod_qmult (U, speye (size (U.H,1)), 1) ;  % convert U to matrix form
V = cod_qmult (V, speye (size (V.H,1)), 1) ;  % convert V to matrix form
display (U) ;
display (V) ;
err = norm (C - U*R*V', 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_cod_sparse'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_cod_sparse'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% dense QR of A'
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense QR factorization of A'':\n') ;
F = factorize (A', [ ], 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
Q = f.Q ;
R = f.R ;
C = F.A ;

err = norm (C' - Q*R, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_qrt_dense'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_qrt_dense'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% sparse QR of A
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse QR factorization:\n') ;
A = sparse (A) ;
F = factorize (A, [ ], 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
R = f.R ;
P = f.P ;
C = F.A ;
err = norm ((C*P)'*(C*P) - R'*R, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_qr_sparse'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_qr_sparse'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% sparse QR of A'
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse QR factorization of A'':\n') ;
F = factorize (A', [ ], 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
R = f.R ;
P = f.P ;
C = F.A ;
err = norm ((P*C)*(P*C)' - R'*R, 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_qrt_sparse'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_qrt_sparse'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% svd
%-------------------------------------------------------------------------------

fprintf ('\n----------SVD factorization:\n') ;
F = factorize (A, 'svd', 1) ;
display (F) ;
Apinv = inverse (F) ;
display (S) ;

[U,S,V] = svd (F) ;
err = norm (A - U*S*V', 2) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_svd'))
    error ('invalid contents') ;
end
if (~(Apinv.is_inverse) || ~isa (Apinv, 'factorization_svd'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% dense LDL
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense LDL factorization:\n') ;
A = rand (3) ;
A = [zeros(3) A ; A' zeros(3)] ;
F = factorize (A, 'ldl', 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
L = f.L ;
D = f.D ;
C = F.A ;
P = f.P ;
err = norm (P'*C*P - L*D*L', 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_ldl_dense'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_ldl_dense'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------
% sparse LDL
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse LDL factorization:\n') ;
A = sparse (A) ;
F = factorize (A, 'ldl', 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;

f = F.Factors ;
L = f.L ;
D = f.D ;
P = f.P ;
C = F.A ;
err = norm (P'*C*P - L*D*L', 1) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, 'factorization_ldl_sparse'))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, 'factorization_ldl_sparse'))
    error ('invalid contents') ;
end

%-------------------------------------------------------------------------------

fprintf ('\nAll disp tests passed, err: %g\n', err) ;
