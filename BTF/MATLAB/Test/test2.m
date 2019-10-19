%TEST2 test script for BTF
% Requires CSparse and UFget
% Example:
%   test2
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5.

% Copyright 2007, Timothy A. Davis, University of Florida

index = UFget ;
f = find (index.nrows == index.ncols) ;

% too much time:
skip = 1514 ;
f = setdiff (f, skip) ;

[ignore i] = sort (index.nnz (f)) ;
f = f (i) ;
nmat = length (f) ;

T0 = zeros (nmat,1) ;
T1 = zeros (nmat,1) ;
Anz = zeros (nmat,1) ;
figure (1) ;
clf
MN = zeros (nmat, 2) ;
Nzdiag = zeros (nmat,1) ;

% warmup
p = maxtrans (sparse (1)) ;		%#ok
p = btf (sparse (1)) ;			%#ok
p = cs_dmperm (sparse (1)) ;		%#ok
a = cs_transpose (sparse (1)) ;		%#ok

for k = 1:nmat

    Prob = UFget (f (k)) ;
    A = Prob.A ;

    Nzdiag (k) = nnz (diag (A)) ;

    [m n] = size (A) ;
    Anz (k) = nnz (A) ;
    MN (k,:) = [m n] ;

    tic
    [p,q,r] = btf (A) ;
    t0 = toc ;
    s0 = sum (q > 0) ;
    T0 (k) = max (1e-9, t0) ;

    tic
    [p2,q2,r2] = cs_dmperm (A) ;
    t1 = toc ;
    s1 = sum (dmperm (A) > 0) ;
    T1 (k) = max (1e-9, t1) ;

    fprintf (...
    '%4d btf %10.6f cs_dmperm %10.6f rel: %8.4f\n', ...
	f(k), t0, t1, t0 / t1) ;

    if (s0 ~= s1)
	error ('!') ;
    end

    C = A (p, abs (q)) ;
    subplot (1,2,1) ;
    cspy (C) ;
    z = find (q < 0) ;
    zd = nnz (diag (C (z,z))) ;
    if (zd > 0)
	error ('?') ;
    end

    minnz = Anz (1) ;
    maxnz = nnz (A) ;

    subplot (1,2,2) ;
    loglog (Anz (1:k), T0 (1:k) ./ T1 (1:k), ...
	'o', [minnz maxnz], [1 1], 'r-') ;
    drawnow

    clear C A Prob
end


