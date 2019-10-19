index = UFget ;

[ignore f] = sort (max (index.nrows, index.ncols)) ;

clf

% f = f (523:end) ;
% f = f ((find (f == 938)):end) ;

for i = f

    Prob = UFget (i,index)
    A = Prob.A ;
    try
	subplot (1,4,1) ; cspy (A) ;
	drawnow
	subplot (1,4,2) ; cspy (A,64) ;
	drawnow
	subplot (1,4,3) ; cs_dmspy (A) ;
	drawnow
	subplot (1,4,4) ; cs_dmspy (A,0) ;
	drawnow
    catch
	;
    end

    [m n] = size (A) ;
    if (m == n && nnz (diag (A)) == n)
	p = cs_dmperm (A) ;
	if (any (p ~= 1:n))
	    error ('!') ;
	end
	[p q r s cc rr] = cs_dmperm (A) ;
	if (any (p ~= q))
	    error ('not sym!') ;
	end
%	nb = length (r)-1 ;
%	if (nb == 1)
%	    if (any (p ~= 1:n))
%		error ('not I!') ;
%	    end
%	end
    end

    drawnow
    % pause % (1) ;

end
