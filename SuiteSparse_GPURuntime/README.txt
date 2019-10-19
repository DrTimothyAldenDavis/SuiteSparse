SuiteSparse_GPURuntime Copyright (c) 2013-2016, Timothy A. Davis,
Sencer Nuri Yeralan, and Sanjay Ranka.  http://www.suitesparse.com

SuiteSparse_GPURuntime provides helper functions for the GPU.

FOR LINUX/UNIX/Mac USERS who want to use the C++ callable library:

    To compile the SuiteSparse_GPURuntime C++ library, in the Unix shell, do:

        cd Lib ; make

    or just 'make' in this directory. Both static (*.a) and shared
    (*.so) libraries are created.

    To install the librari into /usr/local/lib, do 'make install'.
    The include files are not copied into /usr/local/include, since this
    library is currently not meant to be user-callable.  It is used only
    by SuiteSparseQR.

    To remove it from /usr/local/lib, do 'make uninstall'.

See SuiteSparse_GPURuntime/Doc/License.txt for the license.
