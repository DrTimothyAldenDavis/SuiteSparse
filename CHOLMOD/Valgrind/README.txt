Torture test for CHOLMOD, using valgrind.  Requires Linux.
http://www.suitesparse.com

Type "make" to compile and run CHOLMOMD with valgrind.
Every line of CHOLMOD will be exercised, and its results
checked.  The line "All tests passed" should appear in each
output file (*.grind).  Valgrind should report no errors,
and no malloc'd blocks should be in use at exit.

Note that many, many error messages will appear in the
test output itself (tmp/*.out), because all of CHOLMOD's
error handling is checked as well.  These errors are
expected.

To remove all but the source files and output files from
this directory, type "make clean".  To remove all but the
files in the original distribution and the symbolic links,
type "make distclean".  To remove all but the files in
the original distribution, type "make dopurge".
