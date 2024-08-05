ParU/Tcov:  full statement coverage test of ParU.  Linux is required.

ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
All Rights Reserved.
SPDX-License-Identifier: GPL-3.0-or-later

Some matrices are tested multiple times, to ensure they cover all the lines of
code they can.  ParU has some non-deterministic behavior when creating its
parallel tasks for factorizing multiple fronts in parallel, and this can affect
the test coverage.

To run the test coverage, you must first compile and install all of SuiteSparse.
Starting at the top-level of SuiteSparse, do:

    make local
    make install
    cd ParU/Tcov
    make

Files in this folder:

    cov                     run gcov after the tests
    Makefile                compile and run the tests
    mtest.m                 test the matrices in MATLAB
    paru_brutal_test.cpp    brutal ParU test (for out-of-memory conditions)
    paru_cov.hpp            include file for test programs
    paru_c_test.cpp         test ParU C interface
    paru_quick_test.cpp     test ParU C++ interface
    README.txt              this file

