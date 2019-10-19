%TD test script for BTF
% Example:
%   td
% See also trav.

% Copyright 2006, Timothy A. Davis, University of Florida

randn ('state', 0) ;
rand ('state', 0) ;
figure (1)
clf

for trials = 1:1000
    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    d = 0.1 * rand (1) ;
    A = sprandn (m,n,d) ;

    subplot (1,3,1) ;
    spy (A) ;

    pp = dmperm (A) ;

    sprnk = sum (pp > 0) ;

    % finish pp

    if (all (pp) ~= 0)
	% spy (A (pp,:))
    else
	% spy (0)
    end

    fprintf ('sprnk: %d  m %d n %d\n', sprnk, m, n) ;

    [p,q,r,s] = dmperm (A) ;
    C = A (p,q) ;

    subplot (1,3,2) ;
    hold off
    spy (C)
    hold on

    nk = length (r) - 1 ;

    for k = 1:nk
	r1 = r(k) ;
	r2 = r(k+1) ;
	c1 = s(k)  ;
	c2 = s(k+1) ;
	plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;
    end

    [p2,q2,cp,rp] = dp (A,A') ;

    if (any (sort (p2) ~= 1:m))
	error ('p2!') ;
    end

    if (any (sort (q2) ~= 1:n))
	error ('q2!') ;
    end

    if (cp (5) ~= n+1)
	error ('cp!') ;
    end

    if (rp (5) ~= m+1)
	error ('rp!') ;
    end

    C = A (p2,q2) ;

    subplot (1,3,3) ;
    hold off
    spy (C) ;
    hold on

    r1 = rp(1) ;
    r2 = rp(2) ;
    c1 = cp(1)  ;
    c2 = cp(2) ;
    fprintf ('k %d rows %d to %d cols %d to %d\n', k, r1, r2, c1, c2) ;
    plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    r1 = rp(1) ;
    r2 = rp(2) ;
    c1 = cp(2) ;
    c2 = cp(3) ;
    fprintf ('k %d rows %d to %d cols %d to %d\n', k, r1, r2, c1, c2) ;
    plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    B = C (r1:r2-1, c1:c2-1) ;
    if (nnz (diag (B)) ~= size (B,1))
	error ('C1 diag!') ;
    end

    r1 = rp(2) ;
    r2 = rp(3) ;
    c1 = cp(3) ;
    c2 = cp(4) ;
    fprintf ('k %d rows %d to %d cols %d to %d\n', k, r1, r2, c1, c2) ;
    plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    B = C (r1:r2-1, c1:c2-1) ;
    if (nnz (diag (B)) ~= size (B,1))
	error ('C2 diag!') ;
    end

    r1 = rp(3) ;
    r2 = rp(4) ;
    c1 = cp(4) ;
    c2 = cp(5) ;
    fprintf ('k %d rows %d to %d cols %d to %d\n', k, r1, r2, c1, c2) ;
    plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    B = C (r1:r2-1, c1:c2-1) ;
    if (nnz (diag (B)) ~= size (B,1))
	error ('C3 diag!') ;
    end

    r1 = rp(4) ;
    r2 = rp(5) ;
    c1 = cp(4) ;
    c2 = cp(5) ;
    fprintf ('k %d rows %d to %d cols %d to %d\n', k, r1, r2, c1, c2) ;
    plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    drawnow
    % pause

end
