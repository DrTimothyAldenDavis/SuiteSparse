function test9
% test9: test metis, etree, bisect, nesdis
fprintf ('=================================================================\n');
fprintf ('test9: test metis, etree, bisect, nesdis\n') ;

% Prob = UFget ('LPnetlib/lp_qap15') ;
Prob = UFget ('HB/bcsstk15')
A = Prob.A ;
C = A'*A ;
R = A*A' ;

fprintf ('\nmetis:\n') ;
tic ; p0 = metis (R) ; toc
% [pa po] = etree2 (R (p0,p0)) ; sparse (po - (1:size(R,1)))

tic ; p1 = metis (C) ; toc
% [pa po] = etree2 (C (p1,p1)) ; sparse (po - (1:size(C,1)))

tic ; p2 = metis (C, 'sym') ; toc
% [pa po] = etree2 (C (p1,p1)) ; sparse (po - (1:size(C,1)))

tic ; p3 = metis (A, 'row') ; toc
% [pa po] = etree2 (A (p1,:), 'row') ; sparse (po - (1:size(A,1)))

tic ; p4 = metis (A, 'col') ; toc
% [pa po] = etree2 (A (:,p1), 'col') ; sparse (po - (1:size(A,2)))

fprintf ('\nmetis(A):\n') ;
[m n] = size(A) ;
if (m == n && nnz (A-A') == 0)
    tic ; p5 = metis (A) ; toc
    % figure (1)
    % spy (A (p5,p5)) ;
    [ignore q] = etree (A(p5,p5)) ;
    p5post = p5 (q) ;
    % figure (2)
    % spy (A (p5post,p5post)) ;
    lnz0 = sum (symbfact (A (p5,p5)))
end

fprintf ('\namd:\n') ;
if (m == n && nnz (A-A') == 0)
    tic ; z0 = amd (A) ; toc
    lnz = sum (symbfact (A (z0,z0)))
end

fprintf ('\nbisect:\n') ;
tic ; s0 = bisect (R) ; toc
tic ; s1 = bisect (C) ; toc
tic ; s2 = bisect (C, 'sym') ; toc
tic ; s3 = bisect (A, 'row') ; toc
tic ; s4 = bisect (A, 'col') ; toc


fprintf ('\nnested dissection:\n') ;
tic ; [c0 cp0 cmem0] = nesdis (R) ; toc
tic ; [c1 cp1 cmem1] = nesdis (C) ; toc
tic ; [c2 cp2 cmem2] = nesdis (C, 'sym') ; toc
tic ; [c3 cp3 cmem3] = nesdis (A, 'row') ; toc
tic ; [c4 cp4 cmem4] = nesdis (A, 'col') ; toc

fprintf ('\nnested_dissection(A):\n') ;
if (m == n && nnz (A-A') == 0)
    tic ; c5 = nesdis (A) ; toc
    lnz1 = sum (symbfact (A (c5,c5)))
end

fprintf ('test9 passed\n') ;
