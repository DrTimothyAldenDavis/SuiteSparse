# On the Mac, you need to first install Xcode.  Then
# type this command in the Terminal:
#       xcode-select --install
# to ensure that you have the command-line tools installed.
#
# To just compile the library, do:
#       make
#
# To run a demo using the library 
#       cd ../Demo ; make
#
# To install CSparse for all users, do this in the command line:
#       make install
# Once you do this, you can just compile your program with:
#       cc myprogram.c -lcsparse
# where myprogram.c needs to have the line:
#       #include "cs.h"
# See Demo/cs_demo1.c for an example, which can now be compiled with
#       cc cs_demo1.c -o cs_demo1 -lcsparse

LIBRARY = libcsparse
VERSION = 3.1.5

CF = $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -O -fPIC
I = -I../Include
RANLIB = ranlib
ARCHIVE = $(AR) $(ARFLAGS)

# how to build/install shared and static libraries for Mac and Linux:
INSTALL_LIB = /usr/local/lib
INSTALL_INCLUDE = /usr/local/include
ifeq ($(OS),Windows_NT)
    # Windows (untested)
    AR_TARGET = $(LIBRARY).dll
    SO_TARGET =
else
    # Mac or Linux/Unix
    UNAME := $(shell uname)
    AR_TARGET = $(LIBRARY).a
    ifeq ($(UNAME),Darwin)
        # Mac
        SO_PLAIN  = $(LIBRARY).dylib
        SO_TARGET = $(LIBRARY).$(VERSION).dylib
        SO_OPTS   = -dynamiclib  -compatibility_version $(VERSION) \
                    -current_version $(VERSION)
    else
        # Linux and other variants of Unix
        SO_PLAIN  = $(LIBRARY).so
        SO_TARGET = $(LIBRARY).so.$(VERSION)
        SO_OPTS   = -shared -lm -Wl,-soname -Wl,$(SO_TARGET)
    endif
endif

all: $(AR_TARGET) $(SO_TARGET)

CS = cs_add.o cs_amd.o cs_chol.o cs_cholsol.o cs_counts.o cs_cumsum.o \
	cs_droptol.o cs_dropzeros.o cs_dupl.o cs_entry.o \
	cs_etree.o cs_fkeep.o cs_gaxpy.o cs_happly.o cs_house.o cs_ipvec.o \
	cs_lsolve.o cs_ltsolve.o cs_lu.o cs_lusol.o cs_util.o cs_multiply.o \
	cs_permute.o cs_pinv.o cs_post.o cs_pvec.o cs_qr.o cs_qrsol.o \
	cs_scatter.o cs_schol.o cs_sqr.o cs_symperm.o cs_tdfs.o cs_malloc.o \
	cs_transpose.o cs_compress.o cs_usolve.o cs_utsolve.o cs_scc.o \
	cs_maxtrans.o cs_dmperm.o cs_updown.o cs_print.o cs_norm.o cs_load.o \
	cs_dfs.o cs_reach.o cs_spsolve.o cs_ereach.o cs_leaf.o cs_randperm.o

$(CS): ../Include/cs.h Makefile

%.o: ../Source/%.c ../Include/cs.h
	$(CC) $(CF) $(I) -c $<

$(AR_TARGET): $(CS)
	$(ARCHIVE)  $@ $^
	- $(RANLIB) $@

$(SO_TARGET): $(CS)
	$(CC) $(SO_OPTS) $^ -o $@

install: uninstall all
	cp -f $(SO_TARGET) $(INSTALL_LIB)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	cp -f ../Include/cs.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_LIB)/$(SO_PLAIN)
	chmod 644 $(INSTALL_INCLUDE)/cs.h

uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_INCLUDE)/cs.h

clean:
	- $(RM) *.o

purge: distclean

distclean: clean
	- $(RM) *.a *.obj *.dll *.dylib *.so *.so.*

