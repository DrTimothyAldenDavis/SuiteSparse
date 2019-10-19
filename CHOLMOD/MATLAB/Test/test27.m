function test27
% test27: test nesdis with one matrix (HB/west0479)
fprintf ('=================================================================\n');
fprintf ('test27: test nesdis\n') ;

Prob = UFget ('HB/west0479') ;
dg (Prob.A) ;

fprintf ('test27 passed\n') ;
