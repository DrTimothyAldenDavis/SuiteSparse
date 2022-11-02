#-------------------------------------------------------------------------------
# SPEX_Left_LU/Demo/Makefile: compile and run the demos
#-------------------------------------------------------------------------------

# SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
# Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#-------------------------------------------------------------------------------

default: all

SUITESPARSE ?= $(realpath $(CURDIR)/../../..)
include ../../../SuiteSparse_config/SuiteSparse_config.mk

# uncomment for extra error checking:
CFLAGS += -Wall -Wextra -Wpedantic -Werror

CF = $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -O
I = -I../Include -I../Source -I../../../SuiteSparse_config -I../../../COLAMD/Include -I../../../AMD/Include -I../../SPEX_Util/Include -I../../SPEX_Util/Source -I../../SPEX_Util/Lib

LDLIBS += -lm -lsuitesparseconfig -lgmp -lmpfr -lcolamd -lamd -lspexleftlu \
    -lspexutil

all: lib example example2 spexlu_demo
	- ./example
	- ./example2
	- ./spexlu_demo

lib:
	( cd ../Lib ; $(MAKE) )

example: lib example.c Makefile
	$(CC) $(LDFLAGS) $(CF) $(I) -o example example.c $(LDLIBS)

example2: lib example2.c Makefile
	$(CC) $(LDFLAGS) $(CF) $(I) -o example2 example2.c demos.c $(LDLIBS)

spexlu_demo: lib spexlu_demo.c demos.h demos.c Makefile
	$(CC) $(LDFLAGS) $(CF) $(I) -o spexlu_demo spexlu_demo.c demos.c $(LDLIBS)
clean:
	- $(RM) *.o

purge: distclean

distclean: clean
	- $(RM) -r example example2 example3 example4 example5 spexlu_demo *.a *.dSYM *.obj *.dll
