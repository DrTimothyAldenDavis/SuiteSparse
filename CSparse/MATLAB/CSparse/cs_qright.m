function X = cs_qright (V, Beta, p, Y)
%CS_QRIGHT apply Householder vectors on the right.
%   X = cs_qright(V,Beta,p,Y) computes X = Y*P'*H1*H2*...*Hn = Y*Q where Q is
%   represented by the Householder vectors V, coefficients Beta, and
%   permutation p.  p can be [], which denotes the identity permutation.
%   To obtain Q itself, use Q = cs_qright(V,Beta,p,speye(size(V,1))).
%
%   See also CS_QR, CS_QLEFT.

[m n] = size (V) ;
X = Y ;
if (~isempty (p)) X = X (:,p) ; end
for k = 1:n
    X = X - (X * (Beta (k) * V (:,k))) * V (:,k)' ;
end
