matrices = { ...
  'HB/1138_bus', ...
  'HB/494_bus', ...
  'HB/662_bus', ...
  'HB/685_bus', ...
  'Bomhof/circuit_1', ...
  'Bomhof/circuit_2', ...
  'Bomhof/circuit_3', ...
  'Bomhof/circuit_4', ...
  'Grund/meg1', ...
  'Grund/meg4', ...
  'Hamm/add20', ...
  'Hamm/add32', ...
  'Hamm/bcircuit', ...
  'Hamm/hcircuit', ...
  'Hamm/memplus', ...
  'Hamm/scircuit', ...
  'Sandia/oscil_dcop_01', ...
  'Sandia/oscil_trans_01', ...
  'Sandia/fpga_dcop_01', ...
  'Sandia/fpga_trans_01', ...
  'Sandia/adder_dcop_01', ...
  'Sandia/adder_trans_01', ...
  'Sandia/mult_dcop_01', ...
  'done' } ;

% 'Sandia/init_adder1', ...
% 'Sandia/oscil_dcop_57', ...
% 'Sandia/fpga_dcop_51', ...
% 'Sandia/fpga_trans_02', ...
% 'Sandia/adder_dcop_69', ...
% 'Sandia/adder_trans_02', ...
% 'Sandia/mult_dcop_03', ...

matrices = { ...
  'Grund/meg1', ...
  'Hamm/scircuit', ...
  'Sandia/mult_dcop_01', ...
  'done' } ;

% analyze the matrices

for sss = 1:length (matrices)

% matrices
% sss = input ('select: ') ;

	name = matrices {sss}
	if (name (1) == 'd')
	    break
	end

	Problem = UFget (name) ;
	A = Problem.A ;
	clear Problem
	n = size (A,1) ;
	fprintf ('------------ %s n: %d \n', name, n) ;

	% get symmetry, etc, of original matrix
	control = amd ;
	control (1) = n ;	% no dense rows/cols
	[pp, Info1] = amd (A) ;
	fprintf ('original symmetry: %g   orignal nnz(diag(A)) %g\n', ...
	    Info1 (4), Info1(5)) ;

	% find the BTF form 
	[p,q,r] = dmperm (A) ;
	nblocks = length (r) - 1 ;
	if (nblocks == 1)
	    C = A ;
	else
	    C = A (p,q) ;
	end
	fprintf ('# of blocks: %d\n', nblocks) ;

	% find the biggest block
	maxkk = 0 ;
	for k = nblocks:-1:1
	    k1 = r (k) ;
	    k2 = r (k+1) - 1 ;
	    kk = k2 - k1 + 1 ;
	    if (kk > maxkk)
		maxk = k ;
		maxkk = kk ;
	    end
	end
	fprintf ('biggest block: %d of size %d\n', maxk, maxkk) ; 

	% find the symmetry of biggest block
	k = maxk ;
	k1 = r (k) ;
	k2 = r (k+1) - 1 ;
	B = C (k1:k2, k1:k2) ;
	[pp, Info2] = amd (B) ;
	fprintf ('symmetry of biggest block: %g\n', Info2 (4)) ;
	clear pp

	newsize = size (stuff (A), 1) ;

	figure (1)
	clf
	hold on

	subplot (1,3,1) 
	axis off
	hold on
	spy (stuff (A), 'k')
	title ('original matrix') ;
	% xlabel (' ') ;
	% print ('-deps2', '-r50', sprintf ('orig%d', sss)) ;

	% subplot (1,3,2) 
	% figure (2)
	% clf
	% axis off
	% hold on
	% title ('preordering') ;
	[P,Q,R,Lnz,Info] = klua (A) ;
	C = A (P,Q) ;
	% spy (stuff (C), 'k') ;
	% print ('-deps2', '-r50', sprintf ('perm%d', sss)) ;

	nzb = 0 ;
	for k = nblocks:-1:1
	    k1 = R (k) ;
	    k2 = R (k+1) - 1 ;
	    B = C (k1:k2, k1:k2) ;
	    nzb = nzb + nnz (B) ;
	end
	fprintf ('biggest block: %d of size %d\n', maxk, maxkk) ; 
	fprintf ('total nz in diag blocks: %d\n', nzb) ;
%	for k = nblocks:-1:1
%	    k1 = r (k) ;
%	    k2 = r (k+1) - 1 ;
%	    if (k2 > k1)
%		plot ([k1 k2 k2 k1 k1], [k1 k1 k2 k2 k1], 'r') ;
%	    end
%	end
	% xlabel (' ') ;

	% figure (3)
	subplot (1,3,3)
	axis off
	hold on
	title ('factorization') ;
	[P,Q,R,Lnz,Info] = klua (A) ;
	[L,U,Off,Pnum,Rs] = kluf (A, 1e-3, P,Q,R,Lnz,Info) ;
	spy (stuff (L), 'k');
	spy (stuff (U), 'k') ;
	spy (stuff (Off), 'r') ;
	spy (speye (newsize), 'k')
	% xlabel (sprintf ('nz = %d\n', nnz (L) + nnz (U) + nnz (Off) - n)) ;

	% figure (2)
	subplot (1,3,2) 
	axis off
	hold on
	title ('preordering') ;
	C = A (Pnum,Q) ;	    % ----Pnum, after factorization
	spy (stuff (C), 'k') ;
	% spy (stuff (C), 'k') ;
	% print ('-deps2', '-r50', sprintf ('perm%d', sss)) ;
	spy (triu (stuff (Off),1), 'r') ;
	spy (speye (newsize), 'k')

	print ('-depsc2', sprintf ('mat%d', sss)) ;

	% print ('-djpeg99', '-r900', sprintf ('mat%d', sss)) ;
	% print ('-djpeg90', '-r0', sprintf ('mat%d', sss)) ;
	% print ('-djpeg', '-r900', sprintf ('mat%d', sss)) ;
	% cmd = sprintf ('! jpeg2ps mat%d.jpg > mat%d.eps', sss, sss) ;
	% fprintf ('%s\n', cmd) ;
	% eval (cmd) ;

	% print ('-dtiff', '-r0', sprintf ('mat%d', sss)) ;
	% cmd = sprintf ('! tiff2ps mat%d.tif > mat%d.eps', sss, sss) ;
	% fprintf ('%s\n', cmd) ;
	% eval (cmd) ;

%	clear A B C
%	pack
%	p = zeros (1,n) ;
%	E = spones (D) ;
%	for k = nblocks:-1:1
%	    k1 = r (k) ;
%	    k2 = r (k+1) - 1 ;
%	    if (k2 > k1) 
%		[Lk,Uk,pk] = klu (D (k1:k2, k1:k2)) ;
%		p (k1:k2) = pk + (k1-1) ;
%		% if (any ((k1:k2) ~= p (k1:k2)))
%		 %    pack
%		  %   E (k1:k2,:) = E (p (k1:k2), :) ;
%		% end
%		pack
%		E (k1:k2,k1:k2) = Lk + Uk ;
%		clear Lk Uk
%	    else
%		fprintf ('.') ;
%	    end
%	end
%	fprintf ('\n') ;
%	spy (E)
%	for k = nblocks:-1:1
%	    k1 = r (k) ;
%	    k2 = r (k+1) - 1 ;
%	    if (k2 > k1)
%		plot ([k1 k2 k2 k1 k1], [k1 k1 k2 k2 k1], 'r') ;
%	    end
%	end
%	xlabel (' ') ;

	clear P Q R Lnz Info L U Off Pnum
	pack

end
