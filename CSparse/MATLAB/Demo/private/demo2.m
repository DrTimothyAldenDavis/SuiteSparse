function demo2 (C, sym, name)
%DEMO2: solve a linear system using Cholesky, LU, and QR, with various orderings
%
% Example:
%   demo2 (C, 1, 'name of system')
% See also: cs_demo

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

subplot (2,2,1) ; cspy (C) ;
title (name, 'FontSize', 16, 'Interpreter', 'none') ;
[m n] = size (C) ;
[p,q,r,s,cc,rr] = cs_dmperm (C) ;
subplot (2,2,3) ; cs_dmspy (C) ;
subplot (2,2,4) ; cspy (0) ;
subplot (2,2,2) ; cspy (0) ;
drawnow

sprnk = rr (4) - 1 ;
nb = length (r) - 1 ;
ns = sum ((r (2:nb+1) == r (1:nb)+1) & (s (2:nb+1) == s (1:nb)+1)) ;
fprintf ('blocks: %d singletons %d structural rank %d\n', nb, ns, sprnk) ;

if (sprnk ~= sprank (C))
    error ('sprank mismatch!') ;
end

if (sprnk < min (m,n))
    return ;		    % return if structurally singular
end

% the following code is not in the C version of this demo:
if (m == n)
    if (sym)
	try
	    [L,p] = cs_chol (C) ;		%#ok
	    cspy (L+triu(L',1)) ; title ('L+L''') ;
	catch
	    % tol = 0.001 ;
	    [L,U,p,q] = cs_lu (C,0.001) ;	%#ok
	    cspy (L+U-speye(n)) ; title ('L+U') ;
	end
    else
	[L,U,p,q] = cs_lu (C) ;		%#ok
	cspy (L+U-speye(n)) ; title ('L+U') ;
    end
else
    if (m < n)
	[V,beta,p,R,q] = cs_qr (C') ;		%#ok
    else
	[V,beta,p,R,q] = cs_qr (C) ;		%#ok
    end
    cspy (V+R) ; title ('V+R') ;
end
drawnow

% continue with the MATLAB equivalent of the C cs_demo2 program
for order = [0 3]
    if (order == 0 && m > 1000)
	continue ;
    end
    fprintf ('QR    ') ;
    print_order (order) ;
    b = rhs (m) ;	    % compute right-hand-side
    tic ;
    x = cs_qrsol (C, b, order) ;
    fprintf ('time %8.2f ', toc) ;
    print_resid (C, x, b) ;
end

if (m ~= n)
    return ;
end

for order = 0:3
    if (order == 0 && m > 1000)
	continue ;
    end
    fprintf ('LU    ') ;
    print_order (order) ;
    b = rhs (m) ;	    % compute right-hand-side
    tic ;
    x = cs_lusol (C, b, order) ;
    fprintf ('time %8.2f ', toc) ;
    print_resid (C, x, b) ;
end

if (sym == 0)
    return ;
end

for order = 0:1
    if (order == 0 && m > 1000)
	continue ;
    end
    fprintf ('Chol  ') ;
    print_order (order) ;
    b = rhs (m) ;	    % compute right-hand-side
    tic ;
    x = cs_cholsol (C, b, order) ;
    fprintf ('time %8.2f ', toc) ;
    print_resid (C, x, b) ;
end
