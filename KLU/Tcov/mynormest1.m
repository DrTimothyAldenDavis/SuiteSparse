function est = mynormest1(L, U)

n = size (L,1) ;
est = 0 ;
S = zeros (n,1) ;

for k = 1 : 5

    if k == 1
	X = ones(n,1)/n ;
    else
	j = find(abs(X) == max(abs(X))) ;
	j = j(1) ;	%j can be a vector of indices
	X = zeros(n,1) ;
	X(j) = 1 ;  %construct a ej 
    end

    X = U \ (L \ X) ;

    est_old = est ;
    est = sum(abs(X)) ;

    unchanged = 1 ;
    for i = 1:n
	if (X (i) >= 0)
	    s = 1 ;
	else
	    s = -1 ;
	end
	if (s ~= S (i))
	    S (i) = s ;
	    unchanged = 0 ;
	end
    end

    if k > 1 && (est <= est_old || unchanged)
	break ;
    end
    X = S ;
    X = L' \ (U' \ X) ;

    if k > 1
	jnew = find(abs(X) == max(abs(X))) ;
	if (jnew == j)
	    break ;
	end
    end 
end

for k = 1:n
    X(k) = power(-1, k+1) * (1 + ((k-1)/(n-1))) ;
end
X = U\(L\X) ;
est_new = 2 * sum(abs(X)) / (3 * n) ;
if (est_new > est)
    est = est_new ;
end



