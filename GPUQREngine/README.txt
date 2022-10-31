GPUQREngine, Copyright (c) 2013-2022, Timothy A Davis, Sencer Nuri Yeralan,
and Sanjay Ranka.  All Rights Reserved.
SPDX-License-Identifier: GPL-2.0+

http://www.suitesparse.com

GPUQREngine is a gpu-accelerated QR factorization engine supporting
SuiteSparseQR.

To compile the GPUQREngine C++ library, in the Unix shell, type:

    make

in this directory. Both static (*.a) and shared (*.so) libraries are created.

To install the library into /usr/local/lib, do 'make install'.
To install locally in SuiteSparse/lib instead, do:

    make local
    make install

The include files are not copied into /usr/local/include, since this
library is currently not meant to be user-callable.  It is used only
by SuiteSparseQR.

To uninstall it, do 'make uninstall'.

The actual build is done with cmake.  For Windows, do not use 'make' directly.
Instead, import the CMakeLists.txt file into MS Visual Studio.

See GPUQREngine/Doc/License.txt for the license

