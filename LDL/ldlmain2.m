% LDLMAIN2: compile and run the LDLMAIN program (both with and without AMD)

help ldlmain2

if (~isempty (strfind (computer, '64')))
    error ('64-bit version not yet supported') ;
end

input ('Hit enter to compile and run ldlmain (without AMD): ') ;
try
    mex ldlmain.c ldl.c
    ldlmain
catch
    fprintf ('ldlmain mexFunction failed to compile\n') ;
end

if (~ispc)
    input ('Hit enter to compile and run ldlmain (with AMD): ') ;
    try
	s = pwd ;
	cd ('../AMD') ;
	!make
	cd (s) ;
	mex -output ldlamd -I../AMD/Include -I../UFconfig -L../AMD/Lib -DUSE_AMD ldlmain.c -lamd ldl.c
	ldlamd
    catch
	fprintf ('ldlamd mexFunction failed to compile\n') ;
    end
end

