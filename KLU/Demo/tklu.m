
hb = [1:4] ;
att = [283 284 286] ;
bomhof = [370:373] ;
grund = [465 466] ;
hamm = [539:544] ;

% some but not all Sandia circuits
sandia = [ 1112 1168 1169 1055 1105 1106 1107 1108 984 1052 1053 1054 1109 1111 ] ;

% circuits = [hb bomhof grund hamm sandia att ] ;
circuits = [hb bomhof grund hamm sandia] ;	% exclude ATandT matrices

% for i = [ 13 ]

rand ('state', 0) ;

index = UFget ;
[i mat] = sort (index.nnz) ;

% mat = circuits ;

% mat = 544

for i = mat
    Problem = UFget (i) ;
    fprintf ('\n========================================= Matrix: %s\n', Problem.name) ;
    A = Problem.A ;

    [m n] = size (A) ;
    if (m ~= n)
	fprintf ('skip\n') ;
	continue
    end
    if (~isreal (A))
	fprintf ('skip\n') ;
	continue
    end

%    spy (A)
%    drawnow
    b = rand (size (A,1), 1) ;
    % solve Ax=b using KLU stand-alone program

    x = kludemo (A, b) ;
    fprintf ('KLU    resid: %6.2e\n', norm (A*x-b)) ;

    % solve using umfpack
    tic ;
    x = umfpack (A,'\',b) ;
    t = toc ;
    fprintf ('UMF    resid: %6.2e  wall time: %g\n', norm (A*x-b), t) ;
%    tic
%    x = A\b ;
%    t = toc ;
%    fprintf ('MATLAB resid: %6.2e   wallclock time: %g\n', norm (A*x-b), t) ;
pause
end

