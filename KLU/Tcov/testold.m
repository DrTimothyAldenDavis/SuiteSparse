
index = UFget ;

[i mat] = sort (index.nnz) ;

for j = mat

    for transpose = 0:1

	Problem = UFget (j) ;
	fprintf ('------------- Matrix: %s : transpose %d\n', ...
	    Problem.name, transpose) ;
	A = Problem.A ;
	if (transpose)
	    A = A' ;
	end
	[m n] = size (A) ;

	if (~isreal (A))
	    fprintf ('Skip complex matrix\n') ;
	    continue ;
	end

	[L, U, P, Q] = lu (A) ;

	if (m > n)
	    L = [L [ zeros(n,m-n) ; speye(m-n)]] ;
	end

	% try some forward solves
	err = 0 ;
	for trial = 1:50
	    if (trial == 50)
		% try a full vector
		b = sparse (randn (m, 1)) ;
	    elseif (trial == 49)
		% try an all zero vector
		b = sparse (m, 1) ;
	    else
		b = sprandn (m, 1, 0.1) ;
	    end
	    x = L\b ;
	    x2 = lsolve (L, b) ;
	    err = max (err, norm (x - x2, inf) / max (norm (x, inf), 1)) ;
	    if (err > 1e-12)
		error ('inaccurate') ;
	    end
	end
	fprintf ('error: %g\n', err) ;
    end
end

