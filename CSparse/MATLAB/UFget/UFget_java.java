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
    try
    {
      URL url = new URL (u) ;
      InputStream i = url.openStream ();
      OutputStream o = new FileOutputStream (f);
      byte [ ] s = new byte [4096] ;
      int b ;
      while ((b = i.read (s)) != -1)
      {
        o.write (s, 0, b) ;
      }
      i.close ( ) ;
      o.close ( ) ;
    }
    catch (Exception e)
    {
      System.out.println (e) ;
    }
  }
}
