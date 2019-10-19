#-------------------------------------------------------------------------------
# COLAMD Lib/Makefile
#-------------------------------------------------------------------------------

LIBRARY = libcolamd
VERSION = 2.9.2

default: library

include ../../SuiteSparse_config/SuiteSparse_config.mk

I = -I../Include -I../../SuiteSparse_config

INC = ../Include/colamd.h ../../SuiteSparse_config/SuiteSparse_config.h

SRC = ../Source/colamd.c

OBJ = colamd.o colamd_l.o

colamd.o: $(SRC) $(INC)
	$(CC) $(CF) $(I) -c ../Source/colamd.c

colamd_l.o: $(SRC) $(INC)
	$(CC) $(CF) $(I) -c ../Source/colamd.c -DDLONG -o colamd_l.o

# creates libcolamd.a, a C-callable COLAMD library
$(AR_TARGET):  $(OBJ)
	$(ARCHIVE)  $@ $^
	- $(RANLIB) $@ 

$(SO_TARGET): $(OBJ)
	$(CC) $(SO_OPTS) $^ -o $@

ccode: library

library: $(AR_TARGET) $(SO_TARGET)

clean:
	- $(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

install: uninstall $(SO_TARGET)
	$(CP) $(SO_TARGET) $(INSTALL_LIB)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	$(CP) ../Include/colamd.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_LIB)/$(SO_PLAIN)
	chmod 644 $(INSTALL_INCLUDE)/colamd.h

uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_INCLUDE)/colamd.h

