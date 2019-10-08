// MnistTest class for checking student output against validation labels
// Arguments: <student output file> <validation label file>
// Student output file must only contain output labels, one per line, no header lines
// Validation label file has exactly two comment lines, then 
// <no of labels> <digits used>
// then one label per line
//
// Thomas Johansson thomasj@cs.umu.se  2019-10-03

import java.io.File;
import java.util.Scanner;

public class MnistTest
{
   Scanner studentlabelfile;
   Scanner validationlabelfile;   
   
   public static void main(String args[])
   {
//      System.out.println(args.length);
      if (args.length != 2)
      {
         System.out.println("Call: java MnistTest <student label file> <validation label file>");
         System.exit(-1);
      }
      
      MnistTest mnistTest = new MnistTest(args[0], args[1]);
   }
      
   public MnistTest(String studentlabelname, String validationlabelname)
   {
      try
      {
         studentlabelfile = new Scanner(new File(studentlabelname));
         validationlabelfile = new Scanner(new File(validationlabelname));
      }
      catch (Exception ex)
      {
      }

      validationlabelfile.nextLine();
      validationlabelfile.nextLine();
      int nlabels = validationlabelfile.nextInt();
      validationlabelfile.nextLine();                 // skip to end of line
      
      int h = 0;
      
      for (int i = 0; i < nlabels; i++)
      {
         int label = validationlabelfile.nextInt();
         int slabel = studentlabelfile.nextInt();
         if (label == slabel)
            h = h + 1;
      }

//      System.out.printf("Correct: %5.2f", (100.0 * h / nlabels));
      System.out.print("Percentage of correct classifications: " + (100.0 * h) / nlabels);
      System.out.println("% out of " + nlabels + " images");
      
   }
   
}
