ParU/Tcov:  full statement coverage test of ParU.  Linux is required.

ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
All Rights Reserved.
SPDX-License-Identifier: GPL-3.0-or-later

Some matrices are tested multiple times, to ensure they cover all the lines of
code they can.  ParU has some non-deterministic behavior when creating its
parallel tasks for factorizing multiple fronts in parallel, and this can affect
the test coverage.

