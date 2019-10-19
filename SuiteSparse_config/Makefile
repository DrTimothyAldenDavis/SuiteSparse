#-------------------------------------------------------------------------------
# SuiteSparse_config Makefile
#-------------------------------------------------------------------------------

# version of SuiteSparse_config is also version of SuiteSparse meta-package
LIBRARY = libsuitesparseconfig
VERSION = 4.4.7

default: ccode

include SuiteSparse_config.mk

ccode: all

all: $(AR_TARGET) $(SO_TARGET)

library: all

SuiteSparse_config.o: SuiteSparse_config.c SuiteSparse_config.h
	$(CC) $(CF) -c SuiteSparse_config.c

$(AR_TARGET): SuiteSparse_config.o
	$(ARCHIVE) $(AR_TARGET) SuiteSparse_config.o
	$(RANLIB) $(AR_TARGET)

$(SO_TARGET): SuiteSparse_config.o
	$(CC) $(SO_OPTS) $^ -o $@

distclean: purge

purge: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

# install SuiteSparse_config
install: uninstall all
	$(CP) $(SO_TARGET) $(INSTALL_LIB)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	$(CP) SuiteSparse_config.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_LIB)/$(SO_PLAIN)
	chmod 644 $(INSTALL_INCLUDE)/SuiteSparse_config.h

# uninstall SuiteSparse_config
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_INCLUDE)/SuiteSparse_config.h
