This SPQR/Demo directory contains a simple demo of the SuiteSparseQR package.
"make" will compile and run the demo on a set of matrices in the ../Matrix
directory.  The output should look similar to the qrdemo_out.txt file.

README.txt          this file
Makefile            for compiling and running the C++ and C demos

--------------------------------------------------------------------------------

qrdemo.cpp          C++ demo program
qrsimple.cpp        a very simple C++ demo program
qrsimplec.c         a very simple C demo program
qrdemo_out.txt      output of "make" (compiles and tests the 3 codes above)

--------------------------------------------------------------------------------

qrdemo.m            MATLAB equivalent of qrdemo.cpp.  To compare with the C++
                    qrdemo program, type "qrdemo" in the MATLAB command window.
                    You must first compile and install the SuiteSparseQR MATLAB
                    mexFunctions in SPQR/MATLAB (see SPQR/MATLAB/spqr_make.m).

--------------------------------------------------------------------------------

qrdemoc.c           C demo program, compile and test it with "make cdemo"
qrdemoc_out.txt     output of "make cdemo"

--------------------------------------------------------------------------------

GPU-related demos and tests
demo_colamd*.sh     scripts for running the GPU demo with COLAMD
demo_metis*.sh      scripts for running the GPU demo with METIS
go*.m               MATLAB scripts for testing the GPU version
qrdemo_gpu*.cpp     GPU demos
qrdemo_gpu.out      output of qrdemo_gpu.cpp

--------------------------------------------------------------------------------

Timothy A. Davis, http://www.suitesparse.com
