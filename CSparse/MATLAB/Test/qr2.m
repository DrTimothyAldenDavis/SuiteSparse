function [V,Beta,R] = qr_right (A)
[m n] = size (A) ;
V = zeros (m,n) ;
Beta = zeros (1,n) ;
for k = 1:n
    % [v,beta,s] = gallery ('house', A (k:m,k), 2) ;
    [v,beta,s] = house (A (k:m,k)) ;
    V (k:m,k) = v ;
    Beta (k) = beta ;
    A (k:m,k:n) = A (k:m,k:n) - v * (beta * (v' * A (k:m,k:n))) ;
end
R = triu (A) ;
