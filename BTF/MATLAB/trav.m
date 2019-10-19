%TRAV exhaustive test script for BTF
% Requires UFget in the SuiteSparse
% Example:
%   trav
% See also td.

% Copyright 2006, Timothy A. Davis, Univ. of Florida

doplot = 1 ;
dopause = 0 ;
dostrong = 1 ;

index = UFget ;
f = find (index.nrows == index.ncols) ;
[ignore i] = sort (index.nnz (f)) ;
f = f (i) ;
clear i

figure (1)

% short test set: seg faults, lots of blocks, lots of work, and so on:
nasty = [
        % --- various test matrices (no seg fault, quick run time)
    -(1:8)'  % generated matrices
    904 % vanHeukelum/cage3 (5-by-5)
    819 % Simon/raefsky6 (permuted triangular matrix)
	%
        % --- older seg faults:
    264 % HB/west0156, causes older strongcomp_recursive to fail
    824 % TOKAMAK/utm300 (300-by-300), causes older code to fail
    868 % Pothen/bodyy4
        %
        % --- current seg faults:
    290 % Averous/epb3
    983 % Sanghavi/ecl32
    885 % Pothen/tandem_dual
    879 % Pothen/onera_dual
    955 % Schenk_IBMSDS/2D_54019_highK
    957 % Schenk_IBMSDS/3D_51448_3D
    958 % Schenk_IBMSDS/ibm_matrix_2
    912 % vanHeukelum/cage11
    924 % Andrews/Andrews
    960 % Schenk_IBMSDS/matrix-new_3
    862 % Kim/kim1
    544 % Hamm/scircuit
    897 % Norris/torso2
    801 % Ronis/xenon1
     53 % HB/bcsstk31
    958 % Schenk_IBMSDS/matrix_9
    844 % Cunningham/qa8fk
    845 % Cunningham/qa8fk
    821 % Simon/venkat25
    822 % Simon/venkat50
    820 % Simon/venkat01
    812 % Simon/bbmat
    804 % Rothberg/cfd1
     54 % HB/bcsstk32
    913 % vanHeukelum/cage12
    846 % Boeing/bcsstk39
    972 % Schenk_IBMSDS/para-10
    974 % Schenk_IBMSDS/para-5
    975 % Schenk_IBMSDS/para-6
    976 % Schenk_IBMSDS/para-7
    977 % Schenk_IBMSDS/para-8
    978 % Schenk_IBMSDS/para-9
    961 % Schenk_ISEI/barrier2-10
    962 % Schenk_ISEI/barrier2-11
    963 % Schenk_ISEI/barrier2-12
    964 % Schenk_ISEI/barrier2-1
    965 % Schenk_ISEI/barrier2-2
    966 % Schenk_ISEI/barrier2-3
    967 % Schenk_ISEI/barrier2-4
    968 % Schenk_ISEI/barrier2-9
    851 % Chen/pkustk05
    979 % Kamvar/Stanford
    374 % Bova/rma10
	%
        % --- lots of time:
    395 % DRIVCAV/cavity16
    396 % DRIVCAV/cavity17
    397 % DRIVCAV/cavity18
    398 % DRIVCAV/cavity19
    399 % DRIVCAV/cavity20
    400 % DRIVCAV/cavity21
    401 % DRIVCAV/cavity22
    402 % DRIVCAV/cavity23
    403 % DRIVCAV/cavity24
    404 % DRIVCAV/cavity25
    405 % DRIVCAV/cavity26
    1109 % Sandia/mult_dcop_01
    1110 % Sandia/mult_dcop_02
    1111 % Sandia/mult_dcop_03
    376 % Brethour/coater2
    284 % ATandT/onetone2
    588 % Hollinger/mark3jac100
    589 % Hollinger/mark3jac100sc
    452 % Grund/bayer01
    920 % Hohn/sinc12
    590 % Hollinger/mark3jac120
    591 % Hollinger/mark3jac120sc
    809 % Shyy/shyy161
    448 % Graham/graham1
    283 % ATandT/onetone1
    445 % Garon/garon2
    541 % Hamm/bcircuit
    592 % Hollinger/mark3jac140
    593 % Hollinger/mark3jac140sc
    435 % FIDAP/ex40
    912 % Hohn/sinc15
    894 % Norris/lung2
    542 % Hamm/hcircuit
    752 % Mulvey/finan512
    753 % Mulvey/pfinan512
    564 % Hollinger/g7jac180
    565 % Hollinger/g7jac180sc
    566 % Hollinger/g7jac200
    567 % Hollinger/g7jac200sc
    748 % Mallya/lhr34
    749 % Mallya/lhr34c
    922 % Hohn/sinc18
    447 % Goodwin/rim
    807 % Rothberg/struct3
    286 % ATandT/twotone
    982 % Tromble/language
    953 % Schenk_IBMNA/c-73
    890 % Norris/heart1
    750 % Mallya/lhr71
    751 % Mallya/lhr71c
    925 % FEMLAB/ns3Da
    827 % Vavasis/av41092
    931 % FEMLAB/sme3Db
        %
        % --- lots of time, and seg faults:
    285 % ATandT/pre2
	% --- huge matrix, turn off plotting
    940 % Shenk/af_shell1, memory leak in plot, after call to btf, once.
]' ;

% maxtrans_recursive causes a seg fault on these matrices:
skip_list_maxtrans_recursive = 285 ;

% p = dmperm (A) causes a seg fault on these matrices:
skip_list_dmperm = 285 ;

% [p,q,r] = dmperm (A) causes a seg fault on these matrices:
skip_list_dmperm_btf = ...
[    285 879 885 290 955 957 958     924 960         897        959 844 845 ...
 821 822 820     804    913 846 972 974:978 961:968     979     940] ;

% strongcomp (recursive) causes a seg fault on these matrices:
skip_list_strongcomp_recursive     = ...
[983 285 879 885 290 955 957 958 912 924 960 862 544 897 801 53 959 844 845 ...
 821 822 820 812 804 54 913 846 972 974:978 961:968 851     374 940] ;
skip_list_strongcomp_recursive     = ...
[ skip_list_strongcomp_recursive 592 593 752 753 807 286 982 855 566 567 ] ;

% matrices with the largest # of nonzeros in the set (untested)
toobig = [
928   853   852   356   761   368   973   895   805   849   932 ...
803   854   936   802   850   537   856   898   857   859   971   937 ...
914   858   980   896   806   538   863   369   938   860   941   942 ...
943   944   945   946   947   948   915   939   916 ] ;

f = [ -(1:8) f ] ;
% f = nasty ;


nmat = length (f) ;

for matnum = 1:nmat % {

    j = f (matnum) ;

    if (any (j == toobig))
	fprintf ('\n%4d: %3d %s/%s too big\n', ...
	    matnum, j, index.Group{j}, index.Name{j}) ;
	continue ;
    end

    rand ('state', 0) ;

    % clear all unused variables.
    % nothing here is left that is proportional to the matrix size
    clear A p1 p2 p3 q3 r3 match1 match2 match4 pa ra sa qa B C pb rb pc rc
    clear jumble B11 B12 B13 B21 B22 B23 B31 B32 B33 pjumble qjumble ans
    clear c kbad kgood
    % whos
    % pause

    if (j > 0)
	Problem = UFget (j) ;
	name = Problem.name ;
	A = Problem.A ;
	clear Problem
    else
	% construct the jth test matrix
	j = -j ;
	if (j == 1 || j == 2)
	    B11 = UFget ('Grund/b1_ss') ;	    % 7-by-7 diagonal block
	    B11 = B11.A ;
	    B12 = sparse (zeros (7,2)) ;
	    B12 (3,2) = 1 ;
	    B13 = sparse (ones  (7,5)) ;
	    B21 = sparse (zeros (2,7)) ;
	    B22 = sparse (ones  (2,2)) ;	    % 2-by-2 diagonal block
	    B23 = sparse (ones  (2,5)) ;
	    B31 = sparse (zeros (5,7)) ;
	    B32 = sparse (zeros (5,2)) ;
	    B33 = UFget ('vanHeukelum/cage3') ;	    % 5-by-5 diagonal block
	    B33 = B33.A ;
	    A = [ B11 B12 B13 ; B21 B22 B23 ; B31 B32 B33 ] ;
	    name = '(j=1 test matrix)' ;
	end
	if (j == 2)
	    pjumble = [ 10 7 11 1 13 12 8 2 5 14 9 6 4 3 ] ;
	    qjumble = [ 3 14 2 11 1 8 5 7 10 12 4 13 9 6 ] ;
	    A = A (pjumble, qjumble) ;
	    name = '(j=2 test matrix)' ;
	elseif (j == 3)
	    A = sparse (1) ;
	elseif (j == 4)
	    A = sparse (0) ;
	elseif (j == 5)
	    A = sparse (ones (2)) ;
	elseif (j == 6)
	    A = sparse (2,2) ;
	elseif (j == 7)
	    A = speye (2) ;
	elseif (j == 8)
	    A = sparse (2,2) ;
	    A (2,1) = 1 ;
	end
	if (j > 2)
	    full (A)
	end
    end

    [m n] = size (A) ;
    if (m ~= n)
	continue ;
    end
    fprintf ('\n%4d: ', matnum) ;
    fprintf (' =========================== Matrix: %3d %s\n', j, name) ;
    fprintf ('n: %d nz: %d\n', n, nnz (A)) ;

    if (nnz (A) > 6e6)
	doplot = 0 ;
    end

    %---------------------------------------------------------------------------
    % now try maxtrans
    tic
    match1 = maxtrans (A) ;
    t = toc ;
    s1 = sum (match1 > 0) ;
    fprintf ('n-sprank: %d\n', n-s1) ;
    fprintf ('maxtrans:                %8.2f seconds\n', t) ;
    singular = s1 < n ;

    if (doplot)
	% figure (1)
	clf
	subplot (4,3,1)
	spy (A)
	title (name) ;
    end

    p1 = abs (match1) ;
    if (any (sort (p1) ~= 1:n))
	error ('bad perm') ;
    end
    B = A (:,p1) ;

    if (doplot)
	subplot (4,3,4)
	hold off
	spy (B)
	hold on
	badcol = find (match1 < 0) ;
	Junk = sparse (badcol, badcol, ones (length (badcol), 1), n, n) ;
	% if (~isempty (A))
	    % spy (Junk, 'ro') ;
	% end
	title ('maxtrans') ;
    end

    d = nnz (diag (B)) ;
    if (d ~= s1)
	error ('bad sprank') ;
    end
    clear B

    %---------------------------------------------------------------------------
    % try p = dmperm(A)
    skip_dmperm = any (j == skip_list_dmperm) ;

    if (~skip_dmperm)
	tic
	match4 = dmperm (A) ;
	t = toc ;
	fprintf ('p=dmperm(A):             %8.2f seconds\n', t) ;
	s4 = sum (match4 > 0) ;
	singular4 = (s4 < n) ;

	if (doplot)
	    if (~singular4)
		subplot (4,3,6)
		spy (A (match4,:))
		title ('dmperm') ;
	    end
	end
	if (singular ~= singular4)
	    error ('s4?') ; 
	end
	if (s1 ~= s4)
	    error ('bad sprank') ;
	end
    else
	fprintf ('p=dmperm(A): skip\n') ;
    end

    %---------------------------------------------------------------------------
    nblocks = -1 ;
    skip_dmperm_btf = any (j == skip_list_dmperm_btf) ;
    if (~skip_dmperm_btf)
	% get btf form
	tic
	[pa,qa,ra,sa] = dmperm (A) ;
	t = toc ;
	fprintf ('[p,q,r,s]=dmperm(A):     %8.2f seconds\n', t) ;
	nblocks = length (ra) - 1 ;
	fprintf ('nblocks: %d\n', nblocks) ;
	if (~singular4)
	    checkbtf (A, pa, qa, ra) ;
	    if (doplot)
		subplot (4,3,9)
		drawbtf (A, pa, qa, ra)
		title ('dmperm blocks') 
	    end
	end
    else
	fprintf ('[p,q,r,s]=dmperm(A): skip\n') ;
    end

    jumble = randperm (n) ;

    %---------------------------------------------------------------------------
    % try harwell (returns an error for singular matrices, thus the try/catch)

%{
    if (0)
	try
	    tic
	    [p3,q3,r3] = harwell (A) ;
	    t = toc ;
	    fprintf ('harwell (mc21+mc13):     %8.2f seconds  nblocks: %d\n', ...
		t, length (r3)-1) ;
	    checkbtf (A, p3, q3, r3) ;
	    if (doplot)
		subplot (4,3,5)
		% spy (A (p3,q3)) ;
		drawbtf (A, p3, q3, r3) ;
		title ('mc21+mc13') ;
	    end
	    singular3 = 0 ;
	catch
	    % btf detected a singular matrix
	    singular3 = 1 ;
	end
	if (singular ~= singular3)
	    error ('s3?') ; 
	end
    end

    %---------------------------------------------------------------------------
    % now try maxtrans_recursive
    skip_maxtrans_recursive = any (j == skip_list_maxtrans_recursive) ;
    if (~skip_maxtrans_recursive)
	tic
	match2 = maxtrans_recursive (A) ;
	t = toc ;
	fprintf ('maxtrans_recursive:      %8.2f seconds\n', t) ;
	if (any (match1 ~= match2))
	    error ('maxtrans mismatch')
	end
    else
	fprintf ('maxtrans_recursive: (skip)\n') ;
    end

    %---------------------------------------------------------------------------
    % try strongcomp_recursive

    skip_strong = (~dostrong) | any (j == skip_list_strongcomp_recursive) ;

    if (skip_strong)

	fprintf ('strongcomp_recursive skipped\n') ;
	jumble = randperm (n) ;

    else

	%-----------------------------------------------------------------------
	% try strongcomp_recursive on original matrix
	B = A (:,p1) ;
	tic ;
	[pb,rb] = strongcomp_recursive (B) ;
	t = toc ;
	fprintf ('strongcomp_recursive     %8.2f seconds\n', t) ;
	if (~singular && ~skip_dmperm_btf && (length (rb) ~= nblocks+1))
	    warning ('BTF:invalid','rb') ;
	end
	checkbtf (B, pb, pb, rb) ;
	if (doplot)
	    subplot (4,3,7)
	    drawbtf (B, pb, pb, rb) ;
	    title ('strongcomp-recursive') ;
	end

	%-----------------------------------------------------------------------
	% try strongcomp_recursive again, on a randomly jumbled matrix
	jumble = randperm (n) ;
	C = sparse (B (jumble, jumble)) ;
	tic ;
	[pc,rc] = strongcomp_recursive (C) ;
	t = toc ;
	fprintf ('strongcomp_recur (rand)  %8.2f seconds\n', t) ;
	if (~singular && ~skip_dmperm_btf && (length (rc) ~= nblocks+1))
	    warning ('BTF:invalid', 'rc') ;
	end
	checkbtf (C, pc, pc, rc) ;
	if (doplot)
	    subplot (4,3,8)
	    drawbtf (C, pc, pc, rc) ;
	    title ('strongcomp_recur(rand)') ;
	end

	if (length (rc) ~= length (rb))
	    error ('strongcomp_recursive random mismatch') ;
	end

    end
%}

    %---------------------------------------------------------------------------
    % try strongcomp, non-recursive version

	%-----------------------------------------------------------------------
	% try strongcomp on original matrix
	B = A (:,p1) ;
	tic ;
	[pb,rb] = strongcomp (B) ;
	t = toc ;
	fprintf ('strongcomp               %8.2f seconds\n', t) ;
	if (~singular && ~skip_dmperm_btf && (length (rb) ~= nblocks+1))
	    warning ('BTF:invalid', 'rb') ;
	end
	checkbtf (B, pb, pb, rb) ;
	if (doplot)
	    subplot (4,3,10)
	    drawbtf (B, pb, pb, rb) ;
	    title ('strongcomp') ;
	end

%{
	%-----------------------------------------------------------------------
	% try btf on original matrix
	tic ;
	[pw,qw,rw] = btf (A) ;
	t = toc ;
	fprintf ('btf                      %8.2f seconds nblocks %d\n', ...
	    t, length (rw)-1) ;

	if (any (pw ~= pb))
	    error ('pw') ;
	end
	if (any (rw ~= rb))
	    error ('rw') ;
	end
	if (any (abs (qw) ~= p1 (pw)))
	    error ('qw') ;
	end
	c = diag (A (pw,abs (qw))) ;
	if (~singular && ~skip_dmperm_btf && (length (rw) ~= nblocks+1))
	    warning ('BTF:invalid', 'rw') ;
	end
	checkbtf (A, pw, abs (qw), rw) ;

	kbad  = find (qw < 0) ;
	kgood = find (qw > 0) ;
	if (any (c (kbad) ~= 0))
	    error ('kbad') ;
	end
	if (any (c (kgood) == 0))	    %#ok
	    error ('kgood') ;
	end

	if (doplot)
	    subplot (4,3,3)
	    drawbtf (A, pw, abs (qw), rw) ;
	    if (n < 500)
		for k = kbad
		    plot ([k (k+1) (k+1) k k]-.5, [k k (k+1) (k+1) k]-.5, 'r') ;
		end
	    end
	    title ('btf') ;
	end
%}

	%-----------------------------------------------------------------------
	% try [p,q,r] = strongcomp (A, qin) form
	tic
	[pz,qz,rz] = strongcomp (A, match1) ;
	t = toc ;
	fprintf ('[p,q,r]=strongcomp(A,qin)%8.2f seconds\n', t) ;
	if (any (pz ~= pb))
	    error ('pz') ;
	end
	if (any (rz ~= rb))
	    error ('rz') ;
	end
	if (any (abs (qz) ~= p1 (pz)))
	    error ('qz') ;
	end
	c = diag (A (pz,abs (qz))) ;
	if (~singular && ~skip_dmperm_btf && (length (rz) ~= nblocks+1))
	    warning ('BTF:invalid', 'rz') ;
	end
	checkbtf (A, pz, abs (qz), rz) ;

	kbad  = find (qz < 0) ;
	kgood = find (qz > 0) ;
	if (any (c (kbad) ~= 0))
	    error ('kbad') ;
	end
	if (any (c (kgood) == 0))
	    error ('kgood') ;
	end

	if (doplot)
	    subplot (4,3,2)
	    drawbtf (A, pz, abs (qz), rz) ;
	    if (n < 500)
		for k = kbad
		    plot ([k (k+1) (k+1) k k]-.5, [k k (k+1) (k+1) k]-.5, 'r') ;
		end
	    end
	    title ('strongcomp(A,qin)') ;
	end

	%-----------------------------------------------------------------------
	% try strongcomp again, on a randomly jumbled matrix
	C = sparse (B (jumble, jumble)) ;
	tic ;
	[pc,rc] = strongcomp (C) ;
	t = toc ;
	fprintf ('strongcomp       (rand)  %8.2f seconds\n', t) ;
	if (~singular && ~skip_dmperm_btf && (length (rc) ~= nblocks+1))
	    warning ('BTF:invalid', 'rc') ;
	end
	checkbtf (C, pc, pc, rc) ;
	if (doplot)
	    subplot (4,3,12)
	    drawbtf (C, pc, pc, rc) ;
	    title ('strongcomp(rand)') ;
	end

	if (length (rc) ~= length (rb))
	    error ('strongcomp random mismatch') ;
	end

    %---------------------------------------------------------------------------
    if (doplot)
	drawnow
    end

    if (matnum ~= nmat && dopause)
	input ('Hit enter: ') ;
    end

end % }
