function sample()

rand ('state', 0) ;

index = UFget ;
mat = 1:length(index.nnz) ;
[ignore i] = sort (index.nnz (mat)) ;
mat = mat (i) ;
clear i

bomhof = [370:372] ;
grund = [450 451 453 462 464 465 466 467 468] ;
hamm = [539:544] ;
sandia = [ 984 1053 1055 1106 1108 1109 1112 1169 ] ;
rajat = [1185:1198] ;
circuits = [sandia bomhof grund hamm rajat] ;	


for j = circuits 
    Problem = UFget (j) ;
    if (index.isBinary (j))
	continue ;
    end
    name = Problem.name ;

    A = Problem.A ;
    clear Problem ;

    [m n] = size (A) ;
    if (m ~= n)
	continue ;
    end

    b = rand (n,1) ;
    %fprintf ('\n\n=========================== Matrix: %3d %s \n', j, name) ;
    %fprintf ('n: %d nnz(A): %d \n', n, nnz (A)) ;

    opts = [0.1 1.2 1.2 10 1 0 0 0 ] ;
	
    if isreal(A)
	[x info] = klus(A,b,opts, []) ;
    else
	[x info] = klusz(A,b,opts, []) ;
    end
    clear x ;
    %fprintf('\nklu\n') ;
    fill1 = info (31) + info (32) + info(8) ;
    %fprintf('nnz(L+U) %d nzoff %d\n' , fill, info (37)) ;
    
    %umfpack
    %fprintf('\numfpack\n') ;
    [L U P Q] = lu(A) ;
    fill2 = nnz(L) + nnz(U) ;
    %fprintf('nnz(L+U)  %d\n', fill) ;

    %amd
    %fprintf('\ngilbert-peierls\n') ; ;
    Q = amd(A) ;
    [L U P] = lu(A(Q,Q), 0.1) ;
    fill3 = nnz(L) + nnz(U) ;
    %fprintf('nnz(L+U)  %d\n', fill) ;
    
    fprintf('%s (%d) & %d & %d & %d & %d \\\\\n', name, n, nnz(A),...
                fill1, fill2, fill3) ;
end 

