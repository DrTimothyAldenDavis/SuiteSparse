function cs_demo2
% CS_DEMO2: MATLAB version of the CSparse/Demo/cs_demo2.c program.
%   Solves a linear system using Cholesky, LU, and QR, with various orderings.

matrices = {
't1',
'HB/fs_183_1',
'HB/west0067',
'LPnetlib/lp_afiro',
'HB/ash219',
'HB/mbeacxc',
'HB/bcsstk01',
'HB/bcsstk16',
}

for i = 1:length(matrices)
    name = matrices {i} ;
    [C sym] = get_problem (name, 1e-14) ;
    demo2 (C, sym, name) ;
    input ('Hit enter to continue: ') ;
end
