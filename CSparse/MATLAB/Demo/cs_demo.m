function cs_demo
%CS_DEMO: run all CSparse demos.
%   Your current working directory must be CSparse/MATLAB/Demo to run this demo.

help cs_demo
demos = {'cs_demo1', 'cs_demo2', 'cs_demo3', 'ex1', 'ex2', 'ex3' } ;

for i = 1:length(demos)
    fprintf ('\n\n-------------------------------------------------------\n') ;
    help (demos {i}) ;
    input ('Hit enter to continue: ') ;
    eval (demos {i}) ;
end
