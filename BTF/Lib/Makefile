#-------------------------------------------------------------------------------
# BTF Lib/Makefile
#-------------------------------------------------------------------------------

LIBRARY = libbtf
VERSION = 1.2.2

default: all

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

# for testing only:
# TEST = -DTESTING

C = $(CC) $(CF)

INC = ../Include/btf.h ../Include/btf_internal.h

I = -I../Include -I../../SuiteSparse_config

all: library

library: $(AR_TARGET) $(SO_TARGET)

OBJ = btf_order.o btf_maxtrans.o btf_strongcomp.o \
    btf_l_order.o btf_l_maxtrans.o btf_l_strongcomp.o

$(AR_TARGET): $(OBJ)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

$(SO_TARGET): $(OBJ)
	$(CC) $(SO_OPTS) $^ -o $@

$(OBJ): $(INC)

#-------------------------------------------------------------------------------

btf_order.o: ../Source/btf_order.c
	$(C) -c $(I) $< -o $@

btf_maxtrans.o: ../Source/btf_maxtrans.c
	$(C) -c $(I) $< -o $@

btf_strongcomp.o: ../Source/btf_strongcomp.c
	$(C) -c $(I) $< -o $@

#-------------------------------------------------------------------------------

btf_l_order.o: ../Source/btf_order.c
	$(C) -c $(I) -DDLONG $< -o $@

btf_l_maxtrans.o: ../Source/btf_maxtrans.c
	$(C) -c $(I) -DDLONG $< -o $@

btf_l_strongcomp.o: ../Source/btf_strongcomp.c
	$(C) -c $(I) -DDLONG $< -o $@

#-------------------------------------------------------------------------------

install: uninstall $(SO_TARGET)
	$(CP) $(SO_TARGET) $(INSTALL_LIB)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	$(CP) ../Include/btf.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_LIB)/$(SO_PLAIN)
	chmod 644 $(INSTALL_INCLUDE)/btf.h

uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_INCLUDE)/btf.h

#-------------------------------------------------------------------------------

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)
