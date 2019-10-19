function cs_demo3
% CS_DEMO3: MATLAB version of the CSparse/Demo/cs_demo3.c program.
%   Cholesky update/downdate.

matrices = {
'HB/bcsstk01',
'HB/bcsstk16',
}

for i = 1:length(matrices)
    name = matrices {i} ;
    [C sym] = get_problem (name, 1e-14) ;
    demo3 (C, sym, name) ;
    input ('Hit enter to continue: ') ;
end
