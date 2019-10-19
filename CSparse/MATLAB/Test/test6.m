rand ('state', 0)
maxerr = 0 ;
clf
for trial = 1:201
    n = fix (100 * rand (1)) ;
    d = 0.1 * rand (1) ;
    L = tril (sprandn (n,n,d),-1) + sprand (speye (n)) ;
    b = sprandn (n,1,d) ;

    for uplo = 0:1

	if (uplo == 1)
	    % solve Ux=b instead ;
	    L = L' ;
	end

	x = L\b ;
	sr = 1 + cs_reachr (L,b) ;
	sz = 1 + cs_reachr (L,b) ;

	same (sr,sz) ;

	s2 = 1 + cs_reach (L,b) ;

	if (uplo == 0)
	    x3 = cs_lsolve (L,b) ;
	else
	    x3 = cs_usolve (L,b) ;
	end

	spy ([L b x x3])
	drawnow

	s = sort (sr) ;
	[i j xx] = find (x) ;
	[i3 j3 xx3] = find (x3) ;

	if (isempty (i))
	    if (~isempty (s))
		i
		s
		error ('!') ;
	    end
	elseif (any (s ~= i))
	    i
	    s
	    error ('!') ;
	end

	if (isempty (i3))
	    if (~isempty (s))
		i3
		s
		error ('!') ;
	    end
	elseif (any (s ~= sort (i3)))
	    s
	    i3
	    error ('!') ;
	end

	if (any (s2 ~= sr))
	    s2
	    sr
	    error ('!') ;
	end

	err = norm (x-x3,1) ;
	if (err > 1e-12)
	    x
	    x3
	    uplo
	    err
	    error ('!') 
	end

	maxerr = max (maxerr, err) ;

    end

    drawnow
end
maxerr
