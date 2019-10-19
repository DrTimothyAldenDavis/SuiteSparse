#===============================================================================
# RBio/Lib/Makefile: for compiling the RBio library
#===============================================================================

LIBRARY = librbio
VERSION = 2.2.2

default: all

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

C = $(CC) $(CF)

I = -I../Include -I../../SuiteSparse_config

all: library

library: $(AR_TARGET) $(SO_TARGET)

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

INC = ../Include/RBio.h

DL = RBio.o
DI = RBio_i.o
OBJ = $(DI) $(DL)

$(OBJ): $(INC)

$(AR_TARGET): $(OBJ)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

$(SO_TARGET): $(OBJ)
	$(CC) $(SO_OPTS) $^ -o $@

#-------------------------------------------------------------------------------

# 64-bit long version (default)
RBio.o: ../Source/RBio.c
	$(C) -c $(I) $<

# 32-bit int version (not the default)
RBio_i.o: ../Source/RBio.c
	$(C) -c -DINT $(I) $< -o $@

#-------------------------------------------------------------------------------

# install RBio
install: uninstall $(SO_TARGET)
	$(CP) $(SO_TARGET) $(INSTALL_LIB)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	$(CP) ../Include/RBio.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_LIB)/$(SO_PLAIN)
	chmod 644 $(INSTALL_INCLUDE)/RBio.h

# uninstall RBio
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_INCLUDE)/RBio.h

