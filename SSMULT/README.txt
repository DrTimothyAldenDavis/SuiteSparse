SSMULT version 2.0, Mar 25, 2009.  Distributed under the GNU GPL license (see
below).  Copyright (c) 2007-2009, Timothy A. Davis, University of Florida.
SSMULT is also available under other licenses; contact the author for details.
http://www.cise.ufl.edu/research/sparse

SSMULT is a MATLAB toolbox for multiplying two sparse matrices, C = A*B.  It
always uses less memory than the built-in C=A*B (in MATLAB 7.4 or earlier, at
least).  It is typically faster, particularly when A or B are complex.  It is
also much faster when A or B are diagonal or permutations of diagonal matrices.
Requires MATLAB 6.1 or later (it may work on earlier versions; it has been
tested on MATLAB 6.1, 6.5, 7.0.1, 7.0.4, 7.1, 7.2, 7.3, and 7.4).  Works on
32-bit and 64-bit MATLAB.  Either A or B, or both, can be complex.  It should
work in 64-bit Windows, but it has only been tested on Linux for the 64-bit
case.  Only "double" sparse matrices are supported.

SSMULT appears in MATLAB 7.6 and later, but that version of MATLAB does not
pass the transpose/conjugate flags to ssmult.  Instead, (A*B)' (for example) is
computed by multiplying A*B with ssmult, followed by a seperate
conjugate-transpose operation in MATLAB.  This can cause performance
degradation, particularly when computing x'*y when x and y are large sparse
column vectors.

To compile, install, and test, type

    ssmult_install

in the MATLAB command window.  Then edit your path (with pathtool, or
startup.m) to add the SSMULT directory to your MATLAB path.  For more extensive
tests (which require the UFget package) type sstest2 after installing SSMULT.

For best performance, do not use the "lcc" compiler that ships with MATLAB 7.4
(or earlier) for Windows.  Use another compiler instead.  Type "mex -setup" or
"doc mex" for more information.  For Linux/Unix/Mac, edit your mexopts.sh file
and use the option:

    COPTIMFLAGS='-O3 -DNDEBUG'

Note that there is a workaround for a minor "gcc -O" bug (handling floating-
point underflow) in ssmult_template.c.  Use -DNO_GCC_BUG if this bug does not
affect you, and you might slightly increase your performance.

The Results directory contains the result of sstest on various platforms.
The first three are all on the same laptop, an Intel Core Duo (2GHz, 2GB
memory, 2MB of cache):

CoreDuo_Linux.png       MATLAB 7.4, Intel Core Duo, Linux, gcc v4.1, -O3
CoreDuo_MS_lcc.png      MATLAB 7.4, same laptop as above, lcc compiler.
CoreDuo_MS_vc2005.png   MATLAB 7.4, same laptop, MS VC++ 2005 compiler.

Opteron64_Linux.png     MATLAB 7.3, AMD Opteron (64-bit)
Pentium4M_Linux.png     MATLAB 7.3, Pentium 4M, Linux, gcc version 4.1, -O3

These results show that ssmult is always faster for the matrices in this test,
when using gcc -O3 in Linux.  Comparing the CoreDuo_*.png results, you can see
that lcc generates very slow code; these results are on the same laptop.  If
you see that ssmult is slower than C=A*B in sstest, then check your compiler
and its optimization options.  In particular, "lcc" as the default compiler
used by the "mex" function in MATLAB on Windows will lead to poor performance.
In particular, Microsoft provides Visual C++ 2005 Express Edition for free.
Intel's compiler also generates high-quality code.

If you do not have a C compiler for Windows (other than "lcc" provided with
MATLAB) do the following.  Download and install the following from
www.microsoft.com:

    Microsoft Visual C++ 2005 Express Edition
    Microsoft Platform SDK (Windows Server 2003)

Next, install the compiler for use in the MATLAB "mex" command:

    Right click My Computer and select Properties.
    Click the Advanced tab.
    Click the Environment Variables button.
    Create a new environment variable called MSSdk and set its value to the
        path to the Microsoft Platform SDK.  The default location is
        C:\Program Files\Microsoft Platform SDK for Windows Server 2003 R2\
    In MATLAB, type the command "mex -setup".  Select the Microsoft Visual C++
        2005 Express Edition compiler.

--------------------------------------------------------------------------------

SSMULT is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

SSMULT is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this Module; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA  02110-1301, USA.
