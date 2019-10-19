
for n = 2000:2000
    A = sparse (rand (n,n) + n*eye(n)) ;
    b = rand (n,1) ;
    [x, Info] = klus (A, b, [ ], [ ]) ;
    st = Info (10) ;   % analyze time
    ft = Info (38) ;   % factor time
    f2 = Info (61) ;   % refactor time

    fl = (2/3)*n^3 ;

    tic
    x = A\b ;
    t = toc ;

    fprintf ( ...
'n %4d  a %8.2f f %8.2f (%8.1f) r %8.2f (%8.1f)    x=A\\b %8.2f (%8.1f)\n', ...
	n, st, ft, 1e-6*fl/ft, f2, 1e-6*fl/f2, t, 1e-6*fl/t) ;
end

