function [L,U,P] = lu_rightp (A)
n = size (A,1)
P = eye (n) ;
for k = 1:n
    [x,i] = max (abs (A (k:n,k))) ;			      % partial pivoting
    i = i+k-1 ;
    P ([k i],:) = P ([i k], :) ;
    A ([k i],:) = A ([i k], :) ;			      % (6.10), (6.11)
    A (k+1:n,k) = A (k+1:n,k) / A (k,k) ;				% (6.12)
    A (k+1:n,k+1:n) = A (k+1:n,k+1:n) - A (k+1:n,k) * A (k,k+1:n) ;     % (6.9)
end
L = tril (A,-1) + eye (n) ;
U = triu (A) ;
