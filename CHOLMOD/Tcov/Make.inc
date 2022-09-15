# ------------------------------------------------------------------------------
# Tcov/Make.inc
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# CHOLMOD/Tcov Module.  Copyright (C) 2005-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+
# ------------------------------------------------------------------------------

# valgrind is not used
V =

# covall is used
COVER = ./covall

# with test coverage
CF = -O0 -g --coverage
