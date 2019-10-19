#-------------------------------------------------------------------------------
# AMD Lib/Makefile
#-------------------------------------------------------------------------------

LIBRARY = libamd
VERSION = 2.4.2

default: library

include ../../SuiteSparse_config/SuiteSparse_config.mk

library: $(AR_TARGET) $(SO_TARGET)

C = $(CC) $(CF) -I../Include -I../../SuiteSparse_config

#-------------------------------------------------------------------------------
# source files
#-------------------------------------------------------------------------------

AMD = amd_aat amd_1 amd_2 amd_dump amd_postorder amd_post_tree amd_defaults \
	amd_order amd_control amd_info amd_valid amd_preprocess

CONFIG = ../../SuiteSparse_config/SuiteSparse_config.h

INC = ../Include/amd.h ../Include/amd_internal.h $(CONFIG)

#-------------------------------------------------------------------------------
# object files for each version
#-------------------------------------------------------------------------------

AMDI = $(addsuffix .o, $(subst amd_,amd_i_,$(AMD)))
AMDL = $(addsuffix .o, $(subst amd_,amd_l_,$(AMD)))

#-------------------------------------------------------------------------------
# compile each int and long routine (with no real/complex version)
#-------------------------------------------------------------------------------

amd_i_%.o: ../Source/amd_%.c $(INC)
	$(C) -DDINT -c $< -o $@

amd_l_%.o: ../Source/amd_%.c $(INC)
	$(C) -DDLONG -c $< -o $@

#-------------------------------------------------------------------------------
# Create the static and shared libraries (C versions only)
#-------------------------------------------------------------------------------

$(AR_TARGET): $(AMDI) $(AMDL)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

$(SO_TARGET): $(AMDI) $(AMDL)
	$(CC) $(SO_OPTS) $^ -o $@

#-------------------------------------------------------------------------------
# compile the Fortran versions and the libamdf77.a library (static only)
#-------------------------------------------------------------------------------

fortran: libamdf77.a

AMDF77 = amd.o amdbar.o

amd.o: ../Source/amd.f
	$(F77) $(F77FLAGS) -c ../Source/amd.f -o amd.o

amdbar.o: ../Source/amdbar.f
	$(F77) $(F77FLAGS) -c ../Source/amdbar.f -o amdbar.o

libamdf77.a: $(AMDF77)
	$(ARCHIVE) libamdf77.a $^
	- $(RANLIB) libamdf77.a

#-------------------------------------------------------------------------------
# install (shared C version only)
#-------------------------------------------------------------------------------

# install AMD
install: uninstall $(SO_TARGET)
	$(CP) $(SO_TARGET) $(INSTALL_LIB)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	$(CP) ../Include/amd.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_LIB)/$(SO_PLAIN)
	chmod 644 $(INSTALL_INCLUDE)/amd.h

# uninstall AMD
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_INCLUDE)/amd.h

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

clean:
	- $(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)
