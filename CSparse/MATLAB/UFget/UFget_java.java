// function UFget_java.geturl (url, localfile)
// UFget_java.java:  download a URL and save as a file.
//
// Example usage in MATLAB:
//
//	UFget_java.geturl ...
//	('http://www.cise.ufl.edu/research/sparse/mat/HB/ibm32.mat', ...
//	'ibm32.mat') ;
//
// Example usage at the system command line:
//
//	java UFget_java  \
//	    'http://www.cise.ufl.edu/research/sparse/mat/HB/ibm32.mat' \
//	    'ibm32.mat'
//
// To compile:
//
//	javac 
//
// Compiled with java version 1.5.0, on a SUSE Linux Pentium system.
//
// Copyright 2006, Timothy A. Davis

import java.io.* ;
import java.net.* ;

public class UFget_java
{
    public static void main (String [ ] args)

    {
	geturl (args [0], args [1]) ;
    }

    public static void geturl (String u, String f)
    {
	InputStream i = null ;
	OutputStream o = null ;
	try
	{
	    URL url = new URL (u) ;
	    URLConnection conn = url.openConnection ( ) ;
	    i = conn.getInputStream ( ) ;
	    o = new FileOutputStream (f);
	    byte [ ] s = new byte [4096] ;
	    int b ;
	    while ((b = i.read (s)) != -1)
	    {
		o.write (s, 0, b) ;
	    }
	}
	catch (Exception e)
	{
	    System.out.println (e) ;
	}
	finally
	{
	    try
	    {
		if (i != null) i.close ( ) ;
		if (o != null) o.close ( ) ;
	    }
	    catch (IOException ioe)
	    {
	    }
	}
    }
}
