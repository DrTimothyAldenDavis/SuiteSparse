
include ../../UFconfig/UFconfig.mk

linux: distclean
	./DO.linux

sol: distclean
	./DO.solaris

distclean:
	- ( cd .. ; $(MAKE) purge )
	- ( cd ../../AMD ; $(MAKE) purge )
	- $(RM) -r Out/*

purge: distclean

