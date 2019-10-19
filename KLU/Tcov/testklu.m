function testklu ()

rand ('state', 0) ;

index = UFget ;
mat = 1:length(index.nnz) ;
[ignore i] = sort (index.nnz (mat)) ;
mat = mat (i) ;
clear i

%bomhof = [370:373] ;
bomhof = [370:372] ;
%grund = [449:468] ;
grund = [449 468] ;
hamm = [539 540] ;
%sandia = [ 984:1108 1112:1169 ] ;
sandia = [ 984 1108 1112 1169 ] ;
%rajat = [1185:1198] ;
rajat = [1185 1198] ;
circuits = [278 sandia bomhof grund hamm rajat] ;	


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
    fprintf ('\n\n=========================== Matrix: %3d %s \n', j, name) ;
    fprintf ('n: %d nnz(A): %d \n', n, nnz (A)) ;
    utest(A,b) ;
%    pause
end 

