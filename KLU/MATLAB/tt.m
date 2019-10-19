%TT: test klu
% Example:
%   tt

clear all
clear functions

index = UFget ;
f = find (index.nrows == index.ncols & index.isReal) ;
[ignore i] = sort (index.nnz (f)) ;
f = f (i) ;

% f = 274
% f = 101	; % MATLAB condest is poor

nmat = length (f) ;

conds_klu = ones (1,nmat) ;
conds_matlab = ones (1,nmat) ;

for k = 1:nmat
    i = f (k) ;
    try
	Prob = UFget (i,index) ;
	A = Prob.A ;
	% klu (A)
	[L,U,p,q,R,F,r,info] = klu (A) ;
	% info
	rho = lu_normest (R\A(p,q) - F, L, U) ;
	c = condest (A) ;
	rcond = full (min (abs (diag (U))) / max (abs (diag (U)))) ;
	fprintf (...
	'blocks %6d err %8.2e condest %8.2e %8.2e rcond %8.2e %8.2e\n', ...
	length (r)-1, rho, info.condest, c, info.rcond, rcond) ;

	if (info.rcond ~= rcond)
	    fprintf ('!\n') ;
	    pause
	end

	conds_klu (k) =  info.condest ;
	conds_matlab (k) = c ;

    catch
	% fprintf ('failed\n') ;
    end

    plot (1:k, log10 (conds_klu (1:k) ./ conds_matlab (1:k)), 'o') ;
    drawnow
end
