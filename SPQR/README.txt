SPQR (SuiteSparseQR), Copyright (c) 2008-2022, Timothy A Davis.
All Rights Reserved.
SPDX-License-Identifier: GPL-2.0+

The GPU modules in SPQRGPU are under a different copyright:

    SPQRGPU, Copyright (c) 2008-2022, Timothy A Davis, Sanjay Ranka,
    Sencer Nuri Yeralan, and Wissam Sid-Lakhdar, All Rights Reserved.

SuiteSparseQR is a multithreaded, multifrontal, rank-revealing sparse QR
factorization method, with optional GPU acceleration using NVIDIA GPUs.

SuiteSparseQR has support for GPU computing.  See SPQR/Demo/qrdemo_gpu.cpp, and
do 'make gpu' in SPQR/Demo to test it (compare your output with
SPQR/Demo/qrdemo_gpu.out.  For more extensive tests, see SPQR/Demo/go*.m.

NOTE:  you may get an error about a C++-style commment in the C header file,
/usr/local/cude/include/cuComplex.h.  This is a bug in NVIDIA CUDA 7.0.  If
you encounter this error, then either edit the file and delete all lines
beginning with "//", or upgrade to CUDA 7.5.

QUICK START FOR MATLAB USERS (on Windows, Linux, Solaris, or the Mac OS): To
compile and test the MATLAB mexFunctions, do this in the MATLAB command window:

    cd SuiteSparse/SPQR/MATLAB
    spqr_install
    spqr_demo

FOR MORE DETAILS: please see the User Guide in Doc/spqr_user_guide.pdf.

FOR LINUX/UNIX/Mac USERS who want to use the C++ callable library:

    To compile the C++ library and run a short demo, just type this in
        the Unix shell:

        make demo

    FIXME: If you have an NVIDIA GPU, this also compiles
        the GPU accelerated part of SPQR.

    To compile just the SuiteSparseQR C++ library, in the Unix shell, do:

        make
        sudo make install

    To compile and test an exhaustive test, edit the Tcov/Makefile to select
    the LAPACK and BLAS libraries, and then do (in the Unix shell):

        cd Tcov ; make

    Compilation options:

        -DNPARTITION    to compile without METIS (default is to use METIS)

        -DNEXPERT       to compile without the min 2-norm solution option
                        (default is to include the Expert routines)

        -DTIMING        to compile with timing and exact flop counts enabled
                        (default is to not compile with timing and flop counts)


See SPQR/Doc/License.txt for the license.

