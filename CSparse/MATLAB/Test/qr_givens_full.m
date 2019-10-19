function R = qr_givens_full (A)
[m n] = size (A) ;
for i = 2:m
    for k = 1:min(i-1,n)
	A ([k i],k:n) = givens2 (A(k,k), A(i,k)) * A ([k i],k:n) ;
	A (i,k) = 0 ;
    end
end
R = A ;
