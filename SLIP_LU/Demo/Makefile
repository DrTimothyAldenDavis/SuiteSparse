#-------------------------------------------------------------------------------
# SLIP_LU/Demo/Makefile: compile and run the demos
#-------------------------------------------------------------------------------

# SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
# Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
# SLIP_LU/License for the license.

#-------------------------------------------------------------------------------

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

# uncomment for extra error checking:
# CFLAGS += -Wall -Wextra -Wpedantic -Werror

# CF = $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -O
I = -I../Include -I../Source -I../../SuiteSparse_config -I../../COLAMD/Include -I../../AMD/Include

# LDFLAGS = -L../../lib

LDLIBS += -lm -lgmp -lmpfr -lcolamd -lamd -lsliplu
CS = $(LDLIBS)


all: lib example example2 SLIPLU
	- ./example
	- ./example2
	- ./SLIPLU

lib:
	( cd ../Lib ; $(MAKE) )

example: lib example.c Makefile
	$(CC) $(LDFLAGS) $(CF) $(I) -o example example.c $(CS)

example2: lib example2.c Makefile
	$(CC) $(LDFLAGS) $(CF) $(I) -o example2 example2.c demos.c $(CS)

SLIPLU: lib SLIPLU.c demos.h demos.c Makefile
	$(CC) $(LDFLAGS) $(CF) $(I) -o SLIPLU SLIPLU.c demos.c $(CS)
clean:
	- $(RM) *.o

purge: distclean

distclean: clean
	- $(RM) -r example example2 example3 example4 example5 SLIPLU *.a *.dSYM *.obj *.dll
