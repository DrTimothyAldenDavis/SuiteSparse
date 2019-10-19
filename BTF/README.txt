BTF Version 1.0, May 31, 2007, by Timothy A. Davis
Copyright (C) 2004-2007, University of Florida
BTF is also available under other licenses; contact the author for details.
http://www.cise.ufl.edu/research/sparse

BTF is a software package for permuting a matrix into block upper triangular
form.  It includes a maximum transversal algorithm, which finds a permutation
of a square or rectangular matrix so that it has a zero-free diagonal (if one
exists); otherwise, it finds a maximal matching which maximizes the number of
nonzeros on the diagonal.  The package also includes a method for finding the
strongly connected components of a graph.  These two methods together give the
permutation to block upper triangular form.

Requires UFconfig, in the ../UFconfig directory relative to this directory.
KLU relies on this package to permute

To compile the libbtf.a library, type "make".  The compiled library is located
in BTF/Lib/libbtf.a.  Compile code that uses BTF with -IBTF/Include.

Type "make clean" to remove all but the compiled library, and "make distclean"
to remove all files not in the original distribution.

This package does not include a statement coverage test (Tcov directory) or
demo program (Demo directory).  See the KLU package for both.  The BTF package
does include a MATLAB interface, a MATLAB test suite (in the MATLAB/Test
directory), and a MATLAB demo.

See BTF/Include/btf.h for documentation on how to use the C-callable functions.
Use "help btf", "help maxtrans" and "help strongcomp" in MATLAB, for details on
how to use the MATLAB-callable functions.  Additional details on the use of BTF
are given in the KLU User Guide, normally in ../KLU/Doc/KLU_UserGuide.pdf
relative to this directory.

--------------------------------------------------------------------------------

BTF is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This Module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

--------------------------------------------------------------------------------

A full text of the license is in Doc/lesser.txt.  

--------------------------------------------------------------------------------

Files and directories in the BTF package:

    Doc		    documentation and license
    Include	    include files
    Lib		    compiled BTF library
    Makefile	    Makefile for C and MATLAB versions
    MATLAB	    MATLAB interface
    README.txt	    this file
    Source	    BTF source code

./Doc:

    ChangeLog	    changes in BTF
    lesser.txt	    license

./Include:

    btf.h	    primary user include file
    btf_internal.h  internal include file, not for user programs

./Lib:

    Makefile	    Makefile for C library

./MATLAB:

    btf.c	    btf mexFunction
    btf_install.m   compile and install BTF for use in MATLAB
    btf.m	    btf help
    Contents.m	    contents of MATLAB interface
    Makefile	    Makefile for MATLAB functions
    maxtrans.c	    maxtrans mexFunction
    maxtrans.m	    maxtrans help
    strongcomp.c    strongcomp mexFunction
    strongcomp.m    strongcomp help
    Test	    MATLAB test directory

./MATLAB/Test:

    checkbtf.m	    check a BTF ordering
    drawbtf.m	    plot a BTF ordering
    test1.m	    compare maxtrans and cs_dmperm
    test2.m	    compare btf and cs_dmperm
    test3.m	    extensive test (maxtrans, strongcomp, and btf)
    test4b.m	    test btf maxwork option
    test4.m	    test btf maxwork option
    test5.m	    test maxtrans maxwork option

./Source:

    btf_maxtrans.c	btf_maxtrans C function
    btf_order.c		btf_order C function
    btf_strongcomp.c	btf_strongcomp C function

