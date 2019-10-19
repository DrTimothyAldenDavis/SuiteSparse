function test26 (do_metis)
% test26: test logical full and sparse matrices
fprintf ('=================================================================\n');
fprintf ('test26: test logical full and sparse matrices\n') ;

if (nargin < 1)
    do_metis = 1 ;
end

Prob = UFget ('HB/bcsstk01') ;
A = Prob.A ;
p = amd (A) ;
n = size (A,1) ;
A = A (p,p) + 10*speye (n) ;
C = logical (A ~= 0) ;

test26b (A,C) ;
test26b (full (A),C) ;
test26b (full (A), full (C)) ;
test26b (A, full(C)) ;

A = A + 0.001 * (spones (tril (A,-1) + triu (A,1))) * 1i ;

test26b (A,C) ;
test26b (full (A),C) ;
test26b (full (A), full (C)) ;
test26b (A, full(C)) ;
fprintf ('test26 passed\n') ;

    function test26b (A,C)

	p1 = analyze (A) ;
	p2 = analyze (C) ;
	if (any (p1 ~= p2))
	    error ('test 26 failed (analyze)!') ;
	end

	p1 = etree2 (A) ;
	p2 = etree2 (C) ;
	if (any (p1 ~= p2))
	    error ('test 26 failed (etree2)!') ;
	end

	if (do_metis)

	    s1 = bisect (A) ;
	    s2 = bisect (C) ;
	    if (any (s1 ~= s2))
		error ('test 26 failed (bisect)!') ;
	    end

	    p1 = metis (A) ;
	    p2 = metis (C) ;
	    if (any (p1 ~= p2))
		error ('test 26 failed (metis)!') ;
	    end

	    p1 = nesdis (A) ;
	    p2 = nesdis (C) ;
	    if (any (p1 ~= p2))
		error ('test 26 failed (nesdis)!') ;
	    end

	end

	c1 = symbfact2 (A) ;
	c2 = symbfact2 (C) ;
	if (any (c1 ~= c2))
	    error ('test 26 failed (symbfact2)!') ;
	end

	A (1,2) = 0 ;
	A (2,1) = 0 ;
	C = logical (A ~= 0) ;

	L = chol (sparse (A))' ;
	L1 = resymbol (L, A) ;
	L2 = resymbol (L, C) ;
	if (norm (L1 - L2, 1) ~= 0)
	    error ('test 26 failed (resymbol)!') ;
	end

    end

end
