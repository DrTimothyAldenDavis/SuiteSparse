function testRB1
%testRB1: test the RBio toolbox.
%
% Example:
%   testRB1
%
% See also UFget, RBread, RBreade, testRB2.

% Copyright 2006, Timothy A. Davis

files = {
'bcsstk01.rb'
'farm.rb'
'lap_25.pse'
'lap_25.rb'
'west0479.rb'
'west0479.rua' } ;

for k = 1:length(files)
    file = files {k} ;
    fprintf ('%s : ', file) ;
    if (file (end) == 'e')
	[A Z] = RBreade (file) ;
    else
	[A Z] = RBread (file) ;
    end
    fprintf ('%s\n', RBtype (A)) ;
    RBwrite ('temp.rb', A, Z) ;
    [A2 Z2] = RBread ('temp.rb') ;
    if (~isequal (A, A2))
	fprintf ('test failed: %s (A differs)\n', file) ;
	error ('!') ;
    end
    if (~isequal (Z, Z2))
	fprintf ('test failed: %s (Z differs)\n', file) ;
	error ('!') ;
    end
end

load west0479
C = west0479 ;
RBwrite ('mywest', C, 'WEST0479 chemical eng. problem', 'west0479') ;
A = RBread ('mywest') ;
if (~isequal (A, C))
    error ('test failed: west0479 (MATLAB version)') ;
end
fprintf ('west0479 (MATLAB matrix) : %s\n', RBtype (A)) ;

if (~strcmp (RBtype (A), 'rua'))
    error ('test failed: RBtype(A)\n') ;
end
if (~strcmp (RBtype (spones (A)), 'pua'))
    error ('test failed: RBtype(spones(A))\n') ;
end
if (~strcmp (RBtype (2*spones (A)), 'iua'))
    error ('test failed: RBtype(2*spones(A))\n') ;
end
C = A+A' ;
if (~strcmp (RBtype (C), 'rsa'))
    error ('test failed: RBtype(A+A'')\n') ;
end

delete ('mywest') ;
delete ('temp.rb') ;
fprintf ('testRB1: all tests passed\n') ;
