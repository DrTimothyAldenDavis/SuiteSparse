GPUQREngine Copyright (c) 2013, Timothy A. Davis, Sencer Nuri Yeralan,
and Sanjay Ranka.
http://www.suitesparse.com

NOTE: this is version 0.1.0, an alpha release.  Some changes to the API
may occur in the 1.0.0 release.

GPUQREngine is a gpu-accelerated QR factorization engine supporting
SuiteSparseQR.

FOR LINUX/UNIX/Mac USERS who want to use the C++ callable library:

    To compile the GPUQREngine C++ library, in the Unix shell, do:

        make

    Compilation options in SuiteSparse_config/SuiteSparse_config.mk or
    GPUQREngine/*/Makefile

        -DTIMING        to compile with timing and exact flop counts enabled
                        (default is to not compile with timing and flop counts)

--------------------------------------------------------------------------------

GPUQREngine is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

GPUQREngine is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this Module; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA  02110-1301, USA.
