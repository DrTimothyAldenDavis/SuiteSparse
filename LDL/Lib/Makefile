#-------------------------------------------------------------------------------
# LDL Lib/Makefile 
#-------------------------------------------------------------------------------

LIBRARY = libldl
VERSION = 2.2.2

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

I = -I../../SuiteSparse_config -I../Include

C = $(CC) $(CF) $(I)

all: library

library: $(AR_TARGET) $(SO_TARGET)

OBJ = ldl.o ldll.o

ldl.o: ../Source/ldl.c ../Include/ldl.h
	$(C) -c ../Source/ldl.c -o ldl.o

ldll.o: ../Source/ldl.c ../Include/ldl.h
	$(C) -DLDL_LONG -c ../Source/ldl.c -o ldll.o

$(AR_TARGET): $(OBJ)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

$(SO_TARGET): $(OBJ)
	$(CC) $(SO_OPTS) $^ -o $@

distclean: purge

purge: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

# install LDL
install: uninstall $(SO_TARGET)
	$(CP) $(SO_TARGET) $(INSTALL_LIB)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	$(CP) ../Include/ldl.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_LIB)/$(SO_PLAIN)
	chmod 644 $(INSTALL_INCLUDE)/ldl.h

# uninstall LDL
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_INCLUDE)/ldl.h

