function [H,R] = myqr (A)
% function [v,beta,xnorm] = hmake (x)
% function hx = happly (v, beta, x)
[m n] = size (A) ;

H = zeros (m,n) ;
R = zeros (m,n) ;

for k = 1:n

    % apply prior H's
    % fprintf ('\n-----------------init %d\n', k) ;
    x = A (:,k) ;
    for i = 1:k-1
	v = H(((i+1):m),i) ;
	v = [1 ; v] ;
	beta = H (i,i) ;
	% n1 = norm (x (i:m)) ;
	x (i:m) = happly (v, beta, x (i:m)) ;
	% n2 = norm (x (i:m)) ;
	% fprintf ('=============== i %d %g %g\n', i, n1, n2) ;
	% beta
	% v'
	% X = x'
	% pause
	% i
	% x
    end
    % k
    % x

    % make Hk
    % fprintf ('x(k:m) = ') ; x (k:m)
    [v,beta,xnorm] = hmake1 (x (k:m)) ;

    H (k,k) = beta ;
    H (k+1:m, k) = v (2:end) ;

    R (1:(k-1),k) = x (1:(k-1)) ;
    R (k,k) = xnorm ;
    % full (R)
    % pause
end

% s2 = svd (full (R)) ;
% [s1 s2 s1-s2]
