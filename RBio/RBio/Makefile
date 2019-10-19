#===============================================================================
# RBio/MATLAB/Makefile
#===============================================================================

default: all

include ../../UFconfig/UFconfig.mk

I = -I. -I../../UFconfig -I../Include

all: RBread RBwrite RBraw RBtype

MX = $(MEX) $(I)

distclean: purge

purge: clean
	- $(RM) *.mex* *.dll

clean:
	- $(RM) $(CLEAN) private/temp.rb

#===============================================================================

INC = ../Include/RBio.h ../../UFconfig/UFconfig.h

OBJ = RBio.o RBerror.o UFconfig.o

RBerror.o: RBerror.c $(INC)
	$(MX) -c $<

RBio.o: ../Source/RBio.c $(INC)
	$(MX) -c $<

UFconfig.o: ../../UFconfig/UFconfig.c $(INC)
	$(MX) -c $<

RBread: RBread.c $(OBJ)
	$(MX) RBread.c $(OBJ)

RBwrite: RBwrite.c $(OBJ)
	$(MX) RBwrite.c $(OBJ)

RBraw: RBraw.c $(OBJ)
	$(MX) RBraw.c $(OBJ)

RBtype: RBtype.c $(OBJ)
	$(MX) RBtype.c $(OBJ)
