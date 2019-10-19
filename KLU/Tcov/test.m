function test (what)
%TEST: test KLU
% Example:
% what = 2
% test (what)
% what 0: UF circuits
% what 1: sandia matrices only
% what 2: all in UF collection

doplot  = 0 ;
dopause = 0 ;
doplain = 1 ;
%doplain = 0 ;

[tmin maxiters NNNMAX] = control_test ;

fprintf ('\n\nEach code executed for a minimum of %g seconds,\n', tmin) ;
fprintf ('or for a total of %d iterations (whichever comes first)\n',maxiters);
fprintf ('Run times are then divided by the # of iterations performed\n\n') ;

warning off MATLAB:nearlySingularMatrix
warning off MATLAB:singularMatrix
warning off MATLAB:divideByZero

fprintf ('Methods:\n') ;
fprintf ('klu_btf: a single C package that does BTF + AMD + klu.\n') ;
fprintf ('      Always does BTF ordering.  Other codes below can choose.\n') ;
fprintf ('lu_fact: MATLAB wrapper around klu that can do BTF via dmperm.\n') ;
fprintf ('      Same basic algorithm as klu_btf, but not usable outside\n')
fprintf ('      of MATLAB.\n') ;
fprintf ('lu_umfpack: a BTF-capable wrapper for umfpack\n') ;
fprintf ('lu_kundert: a BTF-capable wrapper for Kundert''s Sparse1.3\n\n') ;

other_solvers = 0 ;

rand ('state', 0) ;

index = UFget ;
% mat = find (index.nnz > 1e5)
mat = 1:length(index.nnz) ;
[ignore i] = sort (index.nnz (mat)) ;
mat = mat (i) ;
clear i

symtol = 1e-3 ;
sym = [symtol 1] ;	% symmetric mode (AMD, tolerance of 1e-3)
unsym = [1 0] ;		% unsymmetric mode (COLAMD, tolerance of 1)
unsym_kundert = [0.1 0] ;

do_kundert = 1 ;

hb = 1:4 ;
att = [283 284 286] ;
bomhof = 370:373 ;
grund = [465 466] ;
hamm = 539:544 ;
sandia = [ 984 1053:1055 1105:1112 1168:1169 ] ;
% circuits = [hb bomhof grund hamm att ] ;
circuits = [sandia hb bomhof grund hamm ] ;	% exclude ATandT matrices
%circuits = [1111 544 465] ;	% exclude ATandT matrices

if (nargin < 1)
what = 2
end


if (what == 0)
    % UF circuits
    ufcollection = 1 ;
    ufcircuits = 1 ;
    klu_sym_only = 1 ;
elseif (what == 1)
    % David Day's matrices (sandia list)
    ufcollection = 0 ;
    ufcircuits = 0 ;
    klu_sym_only = 1 ;
elseif (what == 2)
    % all matrices in UF collection
    ufcollection = 1 ;
    ufcircuits = 0 ;
    klu_sym_only = 0 ;
% elseif (what == 3)
%    % J. Hamrle matrices
end

sandia = get_sandia_matrix_list ;

if (ufcollection)
    if (ufcircuits)
	matrix_list = circuits ;
    else
	matrix_list = mat ;
    end
else
    matrix_list = 1 : length (sandia) ; 
end


fprintf ('\n') ;
% fprintf ('NOTE: klu_btf err is norm (A*x-b,inf).  Others are norm(A-L*U,1).\n');
% fprintf ('Thus, you cannot directly compare the "err" result of klu_btf\n') ;
% fprintf ('the other codes\n\n') ;
fprintf ('dmperm time is for MATLAB''s dmperm\n') ;
fprintf ('analyze time (if any) is included in 1st factorization time.\n\n') ;
fprintf ('2nd factorization time does not include analyze time,\n') ; 
fprintf ('and is reported in 2 columns.  The first is with partial pivoting\n');
fprintf ('allowed, and the 2nd column is the time with no pivoting at all.\n');
fprintf ('A dash (-) in either of these columns means the feature is not\n');
fprintf ('supported by the algorithm.\n\n') ;
fprintf ('nzLU is nz (L+U+Off), where Off are the off-diagonal entries\n') ;
fprintf ('\nnoff is the # of off-diagonal pivots chosen by klu_btf.\n') ; 
% for j = mat
% for j = circuits
% for j = 286	% twotone

% matrix_list = 240 % HB/saylr3, singular
% matrix_list = 290 % Averous/epb3
% matrix_list = 88 % HB/bp_400
% matrix_list = 349 % Boeing/bcsstm39

% mm = find (matrix_list == 589) ;
% matrix_list = matrix_list (mm:end) ;

% matrices for which umfpack is better than KLU:
% matrix_list = [ 262 164 144 165 215 231 266 190 193 216 160 154 381 383 380 314 214 429 460 545 546 734 810 336 419 837 422 189 191 43 917 454 385 387 434 384 386 388 568 735 424 547 548 420 918 438 421 437 580 581 432 453 919 412 417 461 390 392 394 570 423 389 391 393 425 410 736 737 444 455 428 582 583 ] ;

% determine if kluf returns the diagonal of U or its inverse
% A = sparse (2) ;
% [P,Q,R,Lnz,Info1] = klua (A)
% [L,U,Off,Pnum,Rs,Info] = kluf (A, P,Q,R,Lnz,Info1)
% if (R == 2)
    % Uinverse = 0 ;
% else
    % Uinverse = 1
    % pause
% end
Uinverse = 1

for j = matrix_list % {

    do_sym_test = 1 ;

    if (ufcollection) 
	Problem = UFget (j) ;
	if (index.isBinary (j))
	    % fprintf ('binary (skip)\n') ;
	    continue ;
	end
	if (strcmp (index.Group {j}, 'Gset'))
	    % fprintf ('Gset (skip)\n') ;
	    continue ;
	end
	if (strcmp (index.Group {j}, 'ATandT'))
	    do_sym_test = 0 ;
	end
	name = Problem.name ;
    else
	name = sandia {j} ;
	Problem = UFget (name) ;
    end

    A = Problem.A ;
    clear Problem ;

    [m n] = size (A) ;
    if (m ~= n)
	% fprintf ('rectangular (skip)\n') ;
	continue ;
    end
    if (~isreal (A))
	% fprintf ('complex (skip)\n') ;
	continue ;
    end

    try
	[p,q,r] = btf (A) ;
    catch
	fprintf ('structurally singular (skip)\n') ;
	continue
    end

%    if (sprank (A) < n)
%	% fprintf ('structurally singular (skip)\n') ;
%	continue
%    end

%% try % {

    symm = nnz (A & A') / nnz (A) ;
    nzd = nnz (diag (A)) ;

    fprintf ('\n\n=========================== Matrix: %3d %s \n', j, name) ;
    fprintf ('n: %d nnz(A): %d  nnz(diag(A)) %d  nnz(diag(A))/n: %g sym: %8.3f\n', ...
	n, nnz (A), nzd, nzd / n, symm) ;

    % get right-hand-side
    b = rand (n,1) .* full (sum (abs (A), 2)) ;

%    x = A\b ;
%    resid = norm (A*x-b, inf) ;
%    cndest = condest (A (:, colamd (A))) ;
%
%    fprintf ('                                                        ') ;
%    fprintf ('condest: %8.2e  x=A\\b resid: %8.2e\n\n', cndest, resid) ;

    fprintf (...
'method            :[ run time in seconds:     ]             err         nzLU flopcount\n') ;

    fprintf (...
'                  :  analyze  1stfact  2ndfact-Piv 2ndfact-NoPiv\n') ;

    fprintf ('-----------------\n') ;

    % try plain klu
    if (doplain)
	q = colamd (A) ;
	tic
	[L,U,p,info] = klu (A, q) ;
	t = toc ;
	fprintf ('plain klu with colamd: err %g time: %8.2f\n', ...
	    lu_normest (A (p,q), L, U), t) ;
    end

    sym_control   = [0 1 10 0 symtol 1.5 1.2 10] ;
    unsym_control = [0 1 10 1 1.0    1.5 1.2 10] ;

%   for transpose = 0:1
    for transpose = 0 % {

	if (transpose)
	    fprintf ('\n------------------- transposed:\n') ;
	    A = A' ;
	end

	%----------------------------------------------------------------------
	% get KLU_BTF flop counts for default options
	%----------------------------------------------------------------------

	[P,Q,R,Lnz,Info1] = klua (A) ;
	[L,U,Off,Pnum,Rs,Info] = kluf (A, P,Q,R,Lnz,Info1) ;
	klu_flop = luflops(L,U) ;

	if (Uinverse)

	    % invert the diagonal of U
	    U = (U')' ;
	    % full (U)
	    % full (triu (U,1))
	    d = full (diag (U)) ;
	    % full (1 ./ d)
	    U = spdiags (1./d, 0, n, n) + triu (U,1) ;
	    % full (U)

	    %  The factorization is L*U + Off = R * (A (Pnum,Q))
	    lu_err = lu_normest ( -Off + (Rs * (A (Pnum,Q))), L, U) ;
	    % fprintf ('klu norm error %g\n', lu_err) ;
	    umax = full (max (abs (diag (U)))) ;
	    umin = full (min (abs (diag (U)))) ;

	else

	    %  The factorization is L*U + Off = R (Pnum,Pnum) \ (A (Pnum,Q))
	    lu_err = lu_normest ( -Off + (Rs \ (A (Pnum,Q))), L, U) ;
	    % fprintf ('klu norm error %g\n', lu_err) ;
	    umax = full (max (abs (diag (U)))) ;
	    umin = full (min (abs (diag (U)))) ;

	end

	myest = umax / umin ;
	fprintf ('my condest umin: %g umax %g condest: %g lu_err %g klu_btf default flop count %8.3e\n\n', umin, umax, myest, lu_err, klu_flop) ;
	if (lu_err > 1e-9)
	    %pause
	end

	%----------------------------------------------------------------------
	% try using a given ordering
	%----------------------------------------------------------------------

	for mangle = 0:3 % {

	if (mangle == 0)
	    Qin = amd (A) ;
	    Pin = Qin ;
	    Qin2 = Qin ;
	    Pin2 = Pin ;
	elseif (mangle == 1)
	    Qin = colamd (A) ;
	    Pin = [ ] ;
	    Qin2 = Qin ;
	    Pin2 = 1:n ;
	elseif (mangle == 2)
	    Qin = [ ] ;
	    Pin = amd (A+A') ;
	    Qin2 = 1:n ;
	    Pin2 = Pin ;
	elseif (mangle == 3)
	    Qin = [ ] ;
	    Pin = [ ] ;
	    Qin2 = 1:n ;
	    Pin2 = 1:n ;
	end

	[P,Q,R,Lnz,Info1] = kluag (A, Pin, Qin) ;
	[L,U,Off,Pnum,Rs,Info] = kluf (A, P,Q,R,Lnz,Info1) ;

	if (Uinverse)

	    % invert the diagonal of U
	    U = (U')' ;
	    % full (U)
	    % full (triu (U,1))
	    d = full (diag (U)) ;
	    % full (1 ./ d)
	    U = spdiags (1./d, 0, n, n) + triu (U,1) ;
	    % full (U)

	    %  The factorization is L*U + Off = R * (A (Pnum,Q))
	    lu_err = lu_normest ( -Off + (Rs * (A (Pnum,Q))), L, U) ;

	else

	    %  The factorization is L*U + Off = R \ (A (Pnum,Q))
	    lu_err = lu_normest ( -Off + (Rs \ (A (Pnum,Q))), L, U) ;

	end

	fprintf ('mangle %d lu_err given: %g\n', mangle, lu_err) ;
	if (lu_err > 1e-9)
	    input ('hi lu error: ') ;
	end

	if (doplot)
	    figure (mangle+1)
	    subplot (2,2,1) ; spy (A) ; title (sprintf ('A mangle:%d',mangle)) ;
	    subplot (2,2,2) ; spy (A (Pin2, Qin2)) ; title ('A(Pin,Qin)') ;
	    subplot (2,2,3) ; drawbtf (A, P, Q, R) ; title ('A(P,Q) btf') ;
	    subplot (2,2,4) ;
		hold off ; spy (L) ; hold on ; spy (U) ; spy (Off,'r') ;
		hold off ;
		title ('L+U+Off') ;
	    drawnow
	end

	end % }

	clear L U Rs Pnum  Q
	pack

	%----------------------------------------------------------------------
	% KLU_BTF with a tolerance of symtol.  Uses BTF and AMD
	%----------------------------------------------------------------------

	% default control
	klu_btf_control = default_control ;

	if (what == 2)
	    orderall = 2 ;
	else
	    orderall = 1 ;
	end

	for ordering = 0:orderall % {
	for do_btf = 0:1 % {
	fprintf ('----\n') ;
	for scale = 0:2 % {

	klu_btf_control (4) = ordering ;
	if (ordering == 1)
	    tol = 1.0 ;
	else
	    tol = symtol ;
	end
	klu_btf_control (5) = tol ;
	klu_btf_control (2) = do_btf ;
	klu_btf_control (3) = scale ;

	for do_harwell = 0:do_btf % {

	if (do_harwell)
	    fprintf ('Har:') ;
	else
	    fprintf ('me::') ;
	end

	st = 0 ;
	ft = 0 ;
	f2 = 0 ;
	iters = 0 ;
	while ((st+ft) < tmin && iters < maxiters)
	    if (do_harwell)
		[x, Info] = klus_harwell (A, b, klu_btf_control, [ ]) ;
	    else
		[x, Info] = klus (A, b, klu_btf_control, [ ]) ;
	    end
	    st = st + Info (10) ;   % analyze time
	    ft = ft + Info (38) ;   % factor time
	    f2 = f2 + Info (61) ;   % refactor time
	    iters = iters + 1 ;
	end
	st = st / iters ;
	ft = ft / iters ;
	f2 = f2 / iters ;

	lunz = Info (31) + Info (32) - n + Info (8) ;
	resid = norm (A*x-b, inf) ;

	fprintf ('klu_btf      ') ;
	if (ordering == 0)
	    fprintf ('  sym:') ;
	else
	    fprintf ('unsym:') ;
	end
	if (do_btf == 0)
	    fprintf ('nobtf') ;
	else
	    fprintf ('  btf') ;
	end
	fprintf (' %9.4f %9.4f %9.4f ', st, st + ft, ft) ;
	if (f2 >= 0)
	    fprintf ('%9.4f', f2) ;
	else
	    fprintf ('     -   ') ;    % refactorization not tested
	end

	fprintf (' lunz: %8d resid: %8.2e offd %6d nzoff %7d nb %6d',...
	    lunz, resid, Info (37), Info (8), Info (4)) ;

	if (ordering == 0)
	    fprintf (' mf: %4.0f %4.0f', 1e-6*klu_flop/ft, 1e-6*klu_flop/f2) ;
	end
	fprintf ('\n') ;

	end % }
	end % }
	end % }
	end % }

	%----------------------------------------------------------------------
	% other solvers
	%----------------------------------------------------------------------

	fprintf ('\n-----------------\n') ;

	if (other_solvers) % {

	    for do_btf = 0:1 % {

		t_dmperm = 0 ;
		if (do_btf == 1)
		    iters = 0 ;
		    while (t_dmperm < tmin && iters < maxiters)
			t = cputime ;
			% [p,q,r] = dmperm (A) ;
			[p,q,r] = btf (A) ;
			t_dmperm = t_dmperm + (cputime-t) ;
			iters = iters + 1 ;
		    end
		    t_dmperm = t_dmperm / iters ;
		    nblocks = length (r) - 1 ;
		    if (nblocks == 1)
			continue
		    end
		    bsize = r (2:nblocks+1) - r (1:nblocks) ;
		    fprintf ('\nBTF: ============= maxblock %d, nblocks: %d, dmperm time %9.4f\n', max (bsize), nblocks, t_dmperm) ;
		end

		fprintf ('::: auto strategy --------------------------\n') ;

		% UMFPACK v4.2+, defaults (uses unsym or sym strategy, as it chooses)
		umf_Control = umfpack ;
		% solver (@lu_umfpack, do_btf, A, t_dmperm, umf_Control, b) ;
		solver (@lu_umfpack, do_btf, A, t_dmperm) ;

		fprintf ('::: unsymmetric strategies -----------------\n') ;

		% ordering only (COLAMD)
		% solver (@lu_order, do_btf, A, t_dmperm) ;

		% MATLAB's LU, defaults
		solver (@lu_matlab, do_btf, A, t_dmperm) ;

		% LU2, defaults (no pruning)
		% solver (@lu_2, do_btf, A, t_dmperm) ;

		% LUprune, defaults (with pruning)
		% solver (@lu_prune, do_btf, A, t_dmperm) ;

		% KLU, unsymmetric mode
		if (~klu_sym_only) % {
		    % solver (@lu_klu, do_btf, A, t_dmperm, unsym) ;

		    % LUfact, unsymmetric mode
%		    solver (@lu_fact, do_btf, A, t_dmperm) ;

		    % Kundert's Sparse1.3, unsymmetric mode
		    if (what < 2 && do_kundert)
			solver (@lu_kundert, do_btf, A, t_dmperm, unsym_kundert) ;
		    end
		end % }

		fprintf ('::: symmetric strategies --------------\n') ;

		if (do_sym_test) % {

		    % ordering only (AMD pure symmetric mode)
		    % solver (@lu_order, do_btf, A, t_dmperm, sym) ;

		    % MATLAB's LU, symmetric mode
		    solver (@lu_matlab, do_btf, A, t_dmperm, sym) ;

		    % LU2, symmetric mode (no pruning)
		    % solver (@lu_2, do_btf, A, t_dmperm, sym) ;

		    % KLU, symmetric mode
		    % solver (@lu_klu, do_btf, A, t_dmperm, sym) ;

		    % LUprune, symmetric mode (with pruning)
		    % solver (@lu_prune, do_btf, A, t_dmperm, sym) ;

		    % LUfact, symmetric mode
		    % solver (@lu_fact, do_btf, A, t_dmperm, sym) ;

		    % Kundert's Sparse1.3, symmetric mode
		    if (what < 2 && do_kundert)
			solver (@lu_kundert, do_btf, A, t_dmperm, sym) ;
		    end

		end % }

	    end % }

	end % }

    end % }

%% catch
%%    fprintf ('failure') ;
%% end % }
    if (dopause)
	pause
    end

end % }

