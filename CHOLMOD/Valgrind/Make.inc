# ------------------------------------------------------------------------------
# CHOLMOD/Valgrind/Makefile
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# CHOLMOD/Valgrind Module.  Copyright (C) 2005-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+
# ------------------------------------------------------------------------------

# valgrind options
V = valgrind --suppressions=suppress --quiet

# covall is not used
COVER =

# no test coverage needed
CF = -O0 -g
