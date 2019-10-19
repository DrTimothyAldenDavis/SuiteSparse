% UMFPACK_TEST: test UMFPACK solve: b/A, A\b with iterative refinement
% Requires the UFsparse package for downloading matrices from the UF
% sparse matrix library.
%
% UMFPACK Version 5.0, Copyright (c) 1995-2006 by Timothy A. Davis.
% All Rights Reserved.  Type umfpack_details for License.

index = UFget ;

f = find (index.nrows == index.ncols) ;
[ignore, i] = sort (index.nrows (f)) ;
f = f (i) ;

Control = umfpack ;
Control (1) = 0 ;

warning ('off', 'all') ;
figure (1)
clf


for i = f

    fprintf ('\nmatrix: %s %s %d\n', index.Group{i}, index.Name{i}, index.nrows(i)) ;

    Prob = UFget (i) ;
    A = Prob.A ;
    n = size (A,1) ;

    b = rand (1,n) ;
    c = b' ;

    try

	%-----------------------------------------------------------------------
	% symbolic factorization
	%-----------------------------------------------------------------------

	[P1, Q1, Fr, Ch, Info] = umfpack (A, 'symbolic') ;
	subplot (2,2,1)
	spy (A)
	title ('A')

	subplot (2,2,2)
	treeplot (Fr (1:end-1,2)') ;
	title ('supercolumn etree')

	%-----------------------------------------------------------------------
	% P(R\A)Q = LU
	%-----------------------------------------------------------------------

	[L,U,P,Q,R,Info] = umfpack (A) ;
	err = lu_normest (P*(R\A)*Q, L, U) ;
	fprintf ('norm est PR\\AQ-LU: %g relative: %g\n', ...
	    err, err / norm (A,1)) ;

	subplot (2,2,3)
	spy (P*A*Q)
	title ('PAQ') ;

	cs = Info (57) ;
	rs = Info (58) ;

	subplot (2,2,4)
	hold off
	spy (L|U)
	hold on
	if (cs > 0)
	    plot ([0 cs n n 0] + .5, [0 cs cs 0 0]+.5, 'c') ;
	end
	if (rs > 0)
	    plot ([0 rs rs 0 0] + cs +.5, [cs cs+rs n n cs]+.5, 'r') ;
	end

	title ('LU factors')
	drawnow

	%-----------------------------------------------------------------------
	% PAQ = LU
	%-----------------------------------------------------------------------

	[L,U,P,Q] = umfpack (A) ;
	err = lu_normest (P*A*Q, L, U) ;
	fprintf ('norm est PAQ-LU:   %g relative: %g\n', ...
	    err, err / norm (A,1)) ;

	%-----------------------------------------------------------------------
	% solve
	%-----------------------------------------------------------------------

	x1 = b/A ;
	y1 = A\c ;
	m1 = norm (b-x1*A) ;
	m2 = norm (A*y1-c) ;

	% factor the transpose
	Control (8) = 2 ;
	[x, info] = umfpack (A', '\', c, Control) ;
	lunz0 = info (44) + info (45) - info (67) ;
	r = norm (A'*x-c) ;

	fprintf (':: %8.2e  matlab: %8.2e %8.2e\n',  r, m1, m2) ;

	% factor the original matrix and solve xA=b
	for ir = 0:4
	    Control (8) = ir ;
	    [x, info] = umfpack (b, '/', A, Control) ;
	    r = norm (b-x*A) ;
	    if (ir == 0)
		lunz1 = info (44) + info (45) - info (67) ;
	    end
	    fprintf ('%d: %8.2e %d %d\n', ir, r, info (81), info (82)) ;
	end

	% factor the original matrix and solve Ax=b
	for ir = 0:4
	    Control (8) = ir ;
	    [x, info] = umfpack (A, '\', c, Control) ;
	    r = norm (A*x-c) ;
	    fprintf ('%d: %8.2e %d %d\n', ir, r, info (81), info (82)) ;
	end

	fprintf ('lunz trans %12d    no trans: %12d  trans/notrans: %10.4f\n', ...
	    lunz0, lunz1, lunz0 / lunz1) ;

	%-----------------------------------------------------------------------
	% get the determinant
	%-----------------------------------------------------------------------

	det1 = det (A) ;
	det2 = umfpack (A, 'det') ;
	[det3 dexp3] = umfpack (A, 'det') ;
	err = abs (det1-det2) ;
	err3 = abs (det1 - (det3 * 10^dexp3)) ;
	denom = det1 ;
	if (denom == 0)
	    denom = 1 ;
	end
	err = err / denom ;
	err3 = err3 / denom ;
	fprintf ('det:  %24.16e + (%24.16e)i MATLAB\n', real(det1), imag(det1)) ;
	fprintf ('det:  %24.16e + (%24.16e)i umfpack\n',real(det2), imag(det2)) ;
	fprintf ('det: (%24.16e + (%24.16e)i) * 10^(%g) umfpack\n', real(det3), imag(det3), dexp3) ;
	fprintf ('diff %g %g\n', err, err3) ;

    catch
	fprintf ('failed\n') ;
    end

%   pause

end
