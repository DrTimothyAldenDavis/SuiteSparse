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

    opts = [0.001 1.2 1.2 10 1 0 0 0 ] ;
	
    if isreal(A)
	[x info] = klus(A,b,opts, []) ;
    else
	[x info] = klusz(A,b,opts, []) ;
    end
    clear x ;
    %fprintf ('analyze time  %10.8f  factor time %10.8f\n',...
    %	      info (10), info(38)) ;
    %fprintf (' refactor time %10.8f solve time %10.8f\n',...
    %	      info (61), info(81)) ;
    
    fprintf ('%s (%d)& %10.4f & %10.4f & %10.4f & %10.4f \\\\\n',...
		name, n, info(10), info(38), info(61), info(81)) ;
    
end 

