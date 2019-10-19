function test3
% test3: test sparse on int8, int16, and logical
fprintf ('=================================================================\n');
fprintf ('test3: test sparse on int8, int16, and logical\n') ;

clear all
c =  ['a' 'b' 0 'd']
sparse(c)
sparse2(c)
sparse(c')
sparse2(c')
whos
nzmax(ans)

try % this will fail
    sparse(int8(c))
catch
    fprintf ('sparse(int8(c)) fails in MATLAB\n') ;
end
sparse2(int8(c))

sparse2 (int16(c))
whos
s = logical(rand(4) > .5)
sparse (s)
whos
sparse2(s)
whos

x = rand(4)
sparse (x > .5)
whos
sparse2 (x > .5)
whos

fprintf ('test3 passed\n') ;
