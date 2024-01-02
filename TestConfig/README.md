SuiteSparse/TestConfig: test `*Config.make` for each package in SuiteSparse

Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
SPDX-License-Identifier: BSD-3-clause

This folder provides a test for each `*Config.cmake` for each package.

To run this test, first build and install all of SuiteSparse.  In the top-level
folder, do:

    make local
    make install

Next, in this directory:

    make

To remove files not in the original distribution, do:

    make clean

