#-------------------------------------------------------------------------------
# RBio Makefile
#-------------------------------------------------------------------------------

VERSION = 2.0.1

default: all

include ../UFconfig/UFconfig.mk

# Compile the C-callable libraries and the Demo programs.
all:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

# Compile the C-callable libraries only.
library:
	( cd Lib ; $(MAKE) )

# Remove all files not in the original distribution
purge:
	( cd RBio ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )

# Remove all files not in the original distribution, except keep the 
# compiled libraries.
clean:
	( cd RBio ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )

distclean: purge

ccode: all

# Compile the MATLAB mexFunctions (you can also use RBinstall.m in MATLAB)
mex:
	( cd RBio ; $(MAKE) )

# Run the test coverage suite.
cov:
	( cd Tcov ; $(MAKE) go )

# Run the test coverage suite using Valgrind.
valgrind:
	( cd Valgrind ; $(MAKE) )

# Compile the C-callable libraries and the Demo programs.
demos:
	( cd Demo ; $(MAKE) )


# install RBio
install:
	$(CP) Lib/librbio.a $(INSTALL_LIB)/librbio.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf librbio.$(VERSION).a librbio.a )
	$(CP) Include/RBio.h $(INSTALL_INCLUDE)

# uninstall RBio
uninstall:
	$(RM) $(INSTALL_LIB)/librbio*.a
	$(RM) $(INSTALL_INCLUDE)/RBio.h

