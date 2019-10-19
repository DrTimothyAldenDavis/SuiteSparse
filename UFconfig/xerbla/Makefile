# Makefile for null-output xerbla

default: ccode

include ../UFconfig.mk

ccode: libcerbla.a

fortran: libxerbla.a 

all: libxerbla.a libcerbla.a

# Fortran version:
libxerbla.a: xerbla.f
	$(F77) $(F77FLAGS) -c xerbla.f
	$(AR) libxerbla.a xerbla.o
	- $(RM) xerbla.o

# C version:
libcerbla.a: xerbla.c xerbla.h
	$(CC) $(CFLAGS) -c xerbla.c
	$(AR) libcerbla.a xerbla.o
	- $(RM) xerbla.o

distclean: purge

purge: clean
	- $(RM) *.o *.a

clean:
	- $(RM) $(CLEAN)
