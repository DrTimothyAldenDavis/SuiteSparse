#-------------------------------------------------------------------------------
# COLAMD Lib/Makefile
#-------------------------------------------------------------------------------

LIBRARY = libcolamd
VERSION = 2.9.3
SO_VERSION = 2

default: library
	$(MAKE) install INSTALL=$(CURDIR)/../.. \
		INSTALL_DOC=$(CURDIR)/../../doc

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

# install COLAMD
install: $(INSTALL_LIB)/$(SO_TARGET)

$(INSTALL_LIB)/$(SO_TARGET): $(SO_TARGET)
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	$(CP) $(SO_TARGET) $(INSTALL_LIB)
	$(SO_INSTALL_NAME) $(INSTALL_LIB)/$(SO_TARGET) $(INSTALL_LIB)/$(SO_TARGET)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_MAIN) )
	$(CP) ../Include/colamd.h $(INSTALL_INCLUDE)
	$(CP) ../README.txt $(INSTALL_DOC)/COLAMD_README.txt
	chmod 755 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_INCLUDE)/colamd.h
	chmod 644 $(INSTALL_DOC)/COLAMD_README.txt

uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_LIB)/$(SO_MAIN)
	$(RM) $(INSTALL_INCLUDE)/colamd.h
	$(RM) $(INSTALL_DOC)/COLAMD_README.txt

