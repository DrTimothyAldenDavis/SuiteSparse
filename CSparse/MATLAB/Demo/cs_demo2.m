function cs_demo2 (do_pause)
% CS_DEMO2: MATLAB version of the CSparse/Demo/cs_demo2.c program.
%   Solves a linear system using Cholesky, LU, and QR, with various orderings.
%
% Example:
%   cs_demo2
% See also: cs_demo

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

matrices = { 't1', 'HB/fs_183_1', 'HB/west0067', 'LPnetlib/lp_afiro', ...
'HB/ash219', 'HB/mbeacxc', 'HB/bcsstk01', 'HB/bcsstk16' } ;

if (nargin < 1)
    do_pause = 1 ;
end

for i = 1:length(matrices)
    name = matrices {i} ;
    [C sym] = get_problem (name, 1e-14) ;
    demo2 (C, sym, name) ;
    if (do_pause)
	input ('Hit enter to continue: ') ;
    end
end
