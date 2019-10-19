clear all
clear functions
cs_test_make	    % compile all CSparse, Demo, Text, and Test mexFunctions

for ktest = 1:28
    fprintf ('\n------------------------ test%d\n', ktest) ;
    eval (sprintf ('test%d', ktest)) ;
end
