rand ('state', 0) ;

index = UFget ;
[i mat] = sort (index.nnz) ;

% [p,q,r,s] = dmperm (A) dies for these matrices.
% all have zero-free diagonals.  Most are irreducible (xenon2 is reducible).
% most are symmetric
bad = [879 885 290 897 844 845 821 822 820 804 913 846 761 895 805 803 802 850 856 898 857 285 859 914 858 896 806 863 369 860 915] ;

% untested, probably fails in dmperm too:
toobig = [916] ; % vanHeukelum/cage15, too big to test

% dies in p = dmperm (A) too
bad2 = [285] ; % ATandT/pre2

if (0)
fprintf ('\nMatrices for which [p,q,r,s] = dmperm (A) dies:\n') ; 
k = 0 ;
for j = bad
    k = k + 1 ;
    fprintf ('%2d: %3d: %s/%s n %d nz %d\n', ...
	k, j, index.Group {j}, index.Name {j}, index.nrows (j), index.nnz (j)) ;
end

fprintf ('\nMatrices for which p = dmperm (A) dies:\n') ; 
k = 0 ;
for j = bad2
    k = k + 1 ;
    fprintf ('%2d: %3d: %s/%s n %d nz %d\n', ...
	k, j, index.Group {j}, index.Name {j}, index.nrows (j), index.nnz (j)) ;
end
end

i = find (mat == (bad (end))) ;
% mat = mat (i:end)

i = find (mat == 462) ;
% mat = mat (i:end) ;

tol = 1e-3 ;

for j = mat

    fprintf ('\n================================================ Matrix %d %s/%s\n', ...
	    j, index.Group {j}, index.Name {j}) ;

    if (any (j == toobig))
	fprintf ('skip (too big)\n') ;
	continue
    end

    Problem = UFget (j) ;
    A = Problem.A ;
    name = Problem.name ;
    clear Problem ;

    [m n] = size (A) ;
    if (m ~= n)
	fprintf ('skip (rectangular)\n') ;
	continue
    end

    if (any (j == bad2))
	fprintf ('skipping sprank\n') ;
    else
	if (sprank (A) < n)
	    fprintf ('skip (structurally singular)\n') ;
	    continue
	end
    end

    figure (1)
    clf
    spy (A)
    title (name) ;
    drawnow

    badone = any (j == bad) ;

    if (badone)
	fprintf ('Skip this matrix for dmperm\n') ;
	t1 = -1 ;
    else
	t = cputime ;
	[p1,q1,r1,s1] = dmperm (A) ;
	t1 = cputime - t ;
	plot_btf (A (p1, q1), r1, 2, n, 'dmperm') ;
    end

    t = cputime ;
    [p,q,r] = btf (A) ;
    t2 = cputime - t ;
    plot_btf (A (p, q), r, 3, n, 'btf') ;

    if (~badone)
	if (length (r) ~= length (r1))
	    error ('r!') ;
	end
    end

    t = cputime ;
    [cp,cq,cr] = charwell (A) ;
    t2 = cputime - t ;
    plot_btf (A (cp, cq), cr, 4, n, 'charwell') ;

    if (length (r) ~= length (cr))
	error ('charwell nblocks') ; 
    end

    if (any (p ~= cp) | any (q ~= cq) | any (r ~= cr))
	error ('charwell') ; 
    end

    % t = cputime ;
    [kp,kq,kr,klnz,kinfo] = klua (A) ;
    % t2 = cputime - t ;
    plot_btf (A (kp, kq), kr, 5, n, 'klua') ;

    if (length (kr) ~= length (cr))
	error ('klua nblocks') ; 
    end

    % t = cputime ;
    [bp,bq,br,blnz] = btfamd (A) ;
    % t2 = cputime - t ;
    plot_btf (A (bp, bq), br, 6, n, 'btfamd') ;

    if (length (kr) ~= length (br))
	error ('btfamd nblocks') ; 
    end

    if (any (blnz ~= klnz))
	blnz
	klnz
	error ('lnz: btfamd vs klua mismatch') ; 
    end

    if (any (bp ~= kp))
	bp
	kp
	error ('p: btfamd vs klua mismatch') ; 
    end

    if (any (bq ~= kq))
	bp
	kp
	error ('q: btfamd vs klua mismatch') ; 
    end

    if (any (br ~= kr))
	br
	kr
	error ('r: btfamd vs klua mismatch') ; 
    end

    if (any (bp ~= kp) | any (bq ~= kq) | any (br ~= kr) | any (blnz ~= klnz))
	error ('btfamd vs klua mismatch') ; 
    end

    fprintf ('                             times %8.2f %8.2f\n', t1, t2) ;

    % kinfo
    try
	[L,U,Off,Pnum,Rs] = kluf (A, tol, kp, kq, kr, klnz, kinfo) ;
	rsdiag = abs (full (diag (Rs))) ;
	rsmax = max (rsdiag) ;
	rsmin = min (rsdiag) ;
	err1 = lu_normest ((Rs (Pnum,Pnum)) \ (A (Pnum,kq)) - Off, L, U) ;
	I = speye (n) ;
	PP = I (Pnum,:) ;
	QQ = I (:,kq) ;
	err2 = lu_normest (PP*(Rs\A)*QQ - Off, L, U) ;

	plot_btf (spones (L) + spones (U), kr, 7, n, 'LU factors of the blocks') ;
	plot_btf (Off, kr, 8, 0, 'offdiagonal entries') ;
    catch
	err1 = -1 ;
	err2 = -1 ;
	rsmin = -1 ;
	rsmax = -1 ;
	fprintf ('kluf failed\n') ;
    end

    try
	B = rand (n, 1) ;
	[X, Info] = klus (A, B, tol) ;
	% Info
	fprintf ('BTF LU err1: %8.2e  max Rs %g min Rs %g\n', err1,rsmax,rsmin);
	fprintf ('BTF LU err2: %8.2e\n', err2) ;
	fprintf ('BTF solve err: %g\n', norm (A*X-B, 1)) ;
    catch
	fprintf ('klus failed\n') ;
    end

    % pause

end
