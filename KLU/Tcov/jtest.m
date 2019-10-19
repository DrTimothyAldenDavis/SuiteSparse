% jtest: test AMD, LU, DMPERM
% Example:
%   jtest

for i = 1:3
    
    if (i == 1)
	load Asmall
    elseif (i == 2)
	load Amiddle
    elseif (i == 3)
	load Alarge
    end

    fprintf ('i %d nnz %d, original matrix:\n', i, nnz (A)) ;
    figure (1)
    spy (A)
    drawnow

    Control = amd ;
    Control (3) = 1 ;

    tic
    [p, Info] = amd (A, Control) ;
    t = toc ;
    fprintf ('AMD ordering time: %g\n', t) ;

    % zero free permutation
    tic
    p = dmperm (A) ;
    t = toc ;
    fprintf ('dmperm ordering time: %g\n', t) ;
    A = A (p,:) ;
    figure (2)
    spy (A)
    title ('after dmperm')
    drawnow

    tic ;
    [p, Info] = amd (A, Control) ;
    t = toc ;
    fprintf ('amd ordering time after dmperm: %g\n', t) ;
    A = A (p,p) ;

    figure (3)
    spy (A)
    title ('after dmperm+amd')
    drawnow

    % factor with MATLAB
    tic
    [L,U,P] = lu (A, 1e-5) ; 
    t = toc ;
    fprintf ('MATLAB GPLU time %g\n', t) ;

    figure (4)
    clf
    hold on
    spy (L) ;
    spy (U) ;
    title ('LU factors') ;
    s = sprintf ('nnz (L): %d   nnz(U): %d\n', nnz (L), nnz (U)) ;
    xlabel (s) ;
    fprintf (s) ;

end

