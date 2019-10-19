% LDLMAIN2: compile and run the LDLMAIN program (both with and without AMD)

help ldlmain2
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
	mex -output ldlamd -I../AMD/Include -L../AMD/Lib -DUSE_AMD ldlmain.c -lamd ldl.c
	ldlamd
    catch
	fprintf ('ldlamd mexFunction failed to compile\n') ;
    end
end

