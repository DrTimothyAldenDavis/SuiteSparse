function X = cs_qleft (V, Beta, p, Y)
%CS_QLEFT apply Householder vectors on the left.
%   X = cs_qleft(V,Beta,p,Y) computes X = Hn*...*H2*H1*P*Y = Q'*Y where Q is
%   represented by the Householder vectors V, coefficients Beta, and
%   permutation p.  p can be [], which denotes the identity permutation.
%
%   See also CS_QR, CS_QRIGHT.

[m2 n] = size (V) ;
[m ny] = size (Y) ;
X = Y ;
if (m2 > m)
    if (issparse (Y))
	X = [X ; sparse(m2-m,ny)] ;
    else
	X = [X ; zeros(m2-m,ny)] ;
    end
end
if (~isempty (p)) X = X (p,:) ; end
for k = 1:n
    X = X - V (:,k) * (Beta (k) * (V (:,k)' * X)) ;
end
