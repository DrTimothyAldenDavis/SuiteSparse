% TP: test KLU
% Example:
%   tp

clear

warning off MATLAB:nearlySingularMatrix

index = UFget ;

[i mat] = sort (index.nnz) ;

if (0)
    jstart = 373 ;
    i = find (mat == jstart) ;
    mat = mat (i:end) ;
    mat = mat (1) ;
end

for j = mat

%  for transpose = 0:1
  for transpose = 0

    Problem = UFget (j) ;
    fprintf ('\n------------- Matrix: %s : transpose %d\n', Problem.name, transpose) ;
    if (index.isBinary (j))
	fprintf ('binary (skip)\n') ;
	continue ;
    end
    if (strcmp (index.Group {j}, 'Gset'))
	fprintf ('Gset (skip)\n') ;
	continue ;
    end
    A = Problem.A ;
    [m n] = size (A) ;
    if (m ~= n)
	fprintf ('rectangular (skip)\n') ;
	continue ;
    end
    if (~isreal (A))
	fprintf ('complex (skip)\n') ;
	continue ;
    end

    if (transpose)
	A = A' ;
    end

    %--------------------------------------------------------------------------
    % umfpack v4.2+
    %--------------------------------------------------------------------------

    t = cputime ;
    tic ;
    %-----------------------------------------
    [L, U, P, Q, R, Info] = umfpack (A) ;
    %-----------------------------------------
    umfpack_time = [toc (cputime-t)] ;

    % scale A for subsequent factorizations, using UMFPACK's scale factors
    A = R\A ;

    umfpack_err = lu_normest (P*A*Q, L, U) ;
    umfpack_nnz = nnz (L) + nnz (U) - n ;
    clear L U P Q

    %--------------------------------------------------------------------------
    % lu: with partial pivoting and colamd
    %--------------------------------------------------------------------------

    t = cputime ;
    tic ;
    %-----------------------------------------
    q = colamd (A, [0.5 0.5 0]) ;
    %-----------------------------------------
    colamd_time = [toc (cputime-t)] ;

    A = A (:,q) ;

    t = cputime ;
    tic ;
    %-----------------------------------------
    [L1, U1, P1] = lu (A) ;
    %-----------------------------------------
    lu_time = colamd_time + [toc (cputime-t)] ;
    lu_err = lu_normest (P1*A, L1, U1) ;
    lu_nnz = nnz (L1) + nnz (U1) - n ;
    clear L1 U1 P1

    %--------------------------------------------------------------------------
    % lu2: with partial pivoting, no pruneing
    %--------------------------------------------------------------------------

    lsize = 5 * nnz (A) ;
    usize = lsize ;
    tol = 1.0 ;
    Control = [tol, lsize, usize] ;

    try
	t = cputime ;
	tic ;
	%-----------------------------------------
	[L2, U2, P2] = lu2 (A, Control) ;
	%-----------------------------------------
	lu2_time = colamd_time + [toc (cputime-t)] ;
	lu2_nnz = nnz (L2 + sparse (n,n)) + nnz (U2 + sparse (n,n)) - n ;
	lu2_err = lu_normest (P2*A, L2, U2) ;
	clear L2 U2 P2
    catch
	fprintf ('lu2: singular, or too big\n') ;
	continue
    end

    %--------------------------------------------------------------------------
    % luprune: with partial pivoting, with pruneing
    %--------------------------------------------------------------------------

    try
	t = cputime ;
	tic ;
	%-----------------------------------------
	[L2, U2, P2] = luprune (A, Control) ;
	%-----------------------------------------
	luprune_time = colamd_time + [toc (cputime-t)] ;
	luprune_nnz = nnz (L2 + sparse (n,n)) + nnz (U2 + sparse (n,n)) - n ;
	luprune_err = lu_normest (P2*A, L2, U2) ;
	clear L2 U2 P2
    catch
	fprintf ('luprune: singular, or too big\n') ;
	continue
    end

    %--------------------------------------------------------------------------
    % symmetric case: min degree of A+A', with pruning
    %--------------------------------------------------------------------------

    fprintf ('symmetry %g nnzdiag/n %g\n', index.pattern_symmetry (j), ...
	index.nnzdiag (j) / n) ;

    do_sym = (index.pattern_symmetry (j) >= 0.7 & index.nnzdiag (j) >= 0.9 * n);

    if (do_sym)
	A = R \ (Problem.A) ;

	t = cputime ;
	tic ;
	%-----------------------------------------
	[q,info] = amd (A) ;
	%-----------------------------------------
	amd_time = [toc (cputime-t)] ;

	lsize = 1.2 * (info (10) + n) ;
	usize = lsize ;
	amd_nnz = 2 * info (10) + n ;
	A = A (q,q) ;
	tol = 0.001 ;
	Control = [tol, lsize, usize] ;

	try
	    t = cputime ;
	    tic ;
	    %-----------------------------------------
	    [L2, U2, P2] = lu2 (A, Control) ;
	    %-----------------------------------------
	    lusym_time = amd_time + [toc (cputime-t)] ;
	    lusym_err = lu_normest (P2*A, L2, U2) ;
	    lusym_nnz = nnz (L2 + sparse (n,n)) + nnz (U2 + sparse (n,n)) - n ;
	    clear L2 U2 P2
	catch
	    fprintf ('sym noprune: singular, or too big\n') ;
	    continue
	end

	try
	    t = cputime ;
	    tic ;
	    %-----------------------------------------
	    [L2, U2, P2] = luprune (A, Control) ;
	    %-----------------------------------------
	    lusymprune_time = amd_time+ [toc (cputime-t)] ;
	    lusymprune_err = lu_normest (P2*A, L2, U2) ;
	    lusymprune_nnz = nnz (L2 + sparse (n,n)) + nnz (U2 + sparse (n,n)) - n ;
	    clear L2 U2 P2
	catch
	    fprintf ('sym prune: singular, or too big\n') ;
	    continue
	end

    end

    %--------------------------------------------------------------------------

    fprintf ('\n-- Final results, Matrix: %s : transpose %d\n', Problem.name, transpose) ;
    fprintf ('UMFPACK    nnz %8d   err %8.2e   wall %8.2f  cpu %8.2f  :: ', ...
	    umfpack_nnz, umfpack_err, umfpack_time) ;
    if (Info (19) == 1)
	fprintf ('unsym\n') ;
    elseif (Info (19) == 2)
	fprintf ('2by2\n') ;
    elseif (Info (19) == 3)
	fprintf ('sym\n') ;
    end

    fprintf ('LU         nnz %8d   err %8.2e   wall %8.2f  cpu %8.2f\n', ...
	lu_nnz, lu_err, lu_time) ; 
    fprintf ('LU2        nnz %8d   err %8.2e   wall %8.2f  cpu %8.2f\n', ...
	lu2_nnz, lu2_err, lu2_time) ; 
    fprintf ('LUprune    nnz %8d   err %8.2e   wall %8.2f  cpu %8.2f\n', ...
	lu2_nnz, lu2_err, luprune_time) ; 
    if (do_sym)
    fprintf ('AMD        nnz %8d\n', amd_nnz) ; 
    fprintf ('LUsym      nnz %8d   err %8.2e   wall %8.2f  cpu %8.2f\n', ...
	lusym_nnz, lusym_err, lusym_time) ; 
    fprintf ('LUsymprune nnz %8d   err %8.2e   wall %8.2f  cpu %8.2f\n', ...
	lusymprune_nnz, lusymprune_err, lusymprune_time) ; 
    end

  end
end

