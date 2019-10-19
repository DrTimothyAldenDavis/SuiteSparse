function test1 (wait)
% test1: test sparse2
fprintf ('=================================================================\n');
fprintf ('test1: test sparse2\n') ;

if (nargin == 0)
    wait = 0 ;
end

m = 3 ;
n = 4 ;

ii = { 1, [2 3]', 2, [ ] } ;
jj = { 1, [2 3]', 3, [ ] } ;
ss = { 1, [2 3]', pi, [ ] } ;

for ki = 1:length (ii)
    for kj = 1:length (jj)
        for ks = 1:length (ss)

            fprintf ('\n-----------------------------------------------\n') ;
            i = ii {ki}
            j = jj {kj}
            s = ss {ks}
	    m
	    n
	    clear A1 A2 B1 B2

	    fprintf ('\nA1 = sparse (i,j,s,m,n)\n') ;
            try % sparse, possibly with invalid inputs
                A1 = sparse (i,j,s,m,n)
		fprintf ('size A1: %d %d\n', size (A1)) ;
            catch
                fprintf ('sparse failed\n') ;
            end

	    fprintf ('\nA2 = sparse2 (i,j,s,m,n)\n') ;
            try % sparse2, possibly with invalid inputs
                A2 = sparse2 (i,j,s,m,n)
		fprintf ('size A2: %d %d\n', size (A2)) ;
            catch
                fprintf ('sparse2 failed\n') ;
            end

	    fprintf ('\nB1 = sparse (i,j,s)\n') ;
            try % sparse, possibly with invalid inputs
                B1 = sparse (i,j,s)
		fprintf ('size B1: %d %d\n', size (B1)) ;
            catch
                fprintf ('sparse failed\n') ;
            end

	    fprintf ('\nB2 = sparse2 (i,j,s)\n') ;
            try % sparse2, possibly with invalid inputs
                B2 = sparse2 (i,j,s)
		fprintf ('size B2: %d %d\n', size (B2)) ;
            catch
                fprintf ('sparse2 failed\n') ;
            end

	    if (wait)
		pause
	    end

        end
    end
end

fprintf ('test1 passed (review the above results)\n') ;
