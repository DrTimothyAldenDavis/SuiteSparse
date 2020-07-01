function gbtest2
%GBTEST2 list all binary operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

optype = gbtest_types ;
opnames = gbtest_binops ;
nbinop = 0 ;

for k1 = 1:length(opnames)

    opname = opnames {k1} ;
    fprintf ('\n=================================== %s\n', opname) ;

    for k2 = 0:length(optype)

        op = opname ;
        if (k2 > 0)
            op = [op '.' optype{k2}] ; %#ok<*AGROW>
        end

        fprintf ('\nop: [%s]\n', op) ;
        try
            if (k2 > 0)
                GrB.binopinfo (op) ;
                nbinop = nbinop + 1 ;
            else
                GrB.binopinfo (op, 'double') ;
            end
        catch
        end
    end
end

fprintf ('\nhelp GrB.binopinfo:\n') ;
GrB.binopinfo ;

fprintf ('number of valid binary operators: %d\n', nbinop) ;
assert (nbinop == 385) ;

fprintf ('gbtest2: all tests passed\n') ;

