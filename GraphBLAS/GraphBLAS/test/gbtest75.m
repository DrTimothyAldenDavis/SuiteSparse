function gbtest75
%GBTEST75 test bitshift

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

fprintf ('\ngbtest75: bitshift\n') ;

for b = -8:8
    fprintf ('.') ;
    for a = intmin ('int8') : intmax ('int8')
        c = bitshift (a, b) ;
        c2 = bitshift (GrB (a), GrB (b)) ;
        assert (isequal (c, c2)) ;
    end
end

fprintf ('\ngbtest75: all tests passed\n') ;

