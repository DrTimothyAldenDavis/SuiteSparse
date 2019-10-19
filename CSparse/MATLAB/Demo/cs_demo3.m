function cs_demo3 (do_pause)
% CS_DEMO3: MATLAB version of the CSparse/Demo/cs_demo3.c program.
%   Cholesky update/downdate.
%
% Example:
%   cs_demo3
% See also: cs_demo

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

matrices = { 'HB/bcsstk01', 'HB/bcsstk16' } ;

if (nargin < 1)
    do_pause = 1 ;
end

for i = 1:length(matrices)
    name = matrices {i} ;
    [C sym] = get_problem (name, 1e-14) ;
    demo3 (C, sym, name) ;
    if (do_pause)
	input ('Hit enter to continue: ') ;
    end
end
