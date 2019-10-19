%test1: KLU test script
% Example:
%   test1
%
% See also klu

% Copyright 2004-2007 Timothy A. Davis, Univ. of Florida
% http://www.cise.ufl.edu/research/sparse

clear all
clear functions
rand ('state', 0) ;

index = UFget ;
f = find (index.nrows == index.ncols & index.isReal) ;
[ignore i] = sort (index.nnz (f)) ;
f = f (i) ;


f = f (1:100) ;

% f = 274
% f = 101	; % MATLAB condest is poor

nmat = length (f) ;

conds_klu = ones (1,nmat) ;
conds_matlab = ones (1,nmat) ;

for k = 1:nmat
    i = f (k) ;
    try
	c = -1 ;
	blocks = 0 ;
	rho = 0 ;
	c2 = 0 ;
	r1 = 0 ;
	r2 = 0 ;
	err = 0 ;

	Prob = UFget (i,index) ;
	A = Prob.A ;
	c = condest (A) ;
	% klu (A)
	% [L,U,p,q,R,F,r,info] = klu (A) ;

	[LU, info, c2] = klu (A) ;

	L = LU.L ;
	U = LU.U ;
	p = LU.p ;
	q = LU.q ;
	R = LU.R ;
	F = LU.F ;
	r = LU.r ;
	blocks = length (r) - 1 ;

	n = size (A,1) ;
	b = rand (n,1) ;
	x = klu (LU,'\',b) ;
	err = norm (A*x-b,1) / norm (A,1) ;

	% info
	rho = lu_normest (R\A(p,q) - F, L, U) ;
	r1 = info.rcond ;
	r2 = full (min (abs (diag (U))) / max (abs (diag (U)))) ;

	if (r1 ~= r2)
	    fprintf ('!\n') ;
	    pause
	end

	conds_klu (k) = c2 ;
	conds_matlab (k) = c ;

    catch
	disp (lasterr) ;
    end

    fprintf (...
'blocks %6d err %8.2e condest %8.2e %8.2e rcond %8.2e %8.2e err %8.2e\n', ...
    blocks, rho, c2, c, r1, r2, err) ;

end
k = nmat ;
plot (1:k, log10 (conds_klu (1:k) ./ conds_matlab (1:k)), 'o') ;
drawnow
