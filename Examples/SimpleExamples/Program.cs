using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CLAP;

namespace SimpleExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                new Parser<Examples>().Run(args, new Examples());
                Console.WriteLine("Done.");
#if DEBUG
                Console.ReadKey();
#endif
            }
            catch (MissingRequiredArgumentException e)
            {
                Console.WriteLine($"Error while trying to execute [{e.Method.MethodInfo.Name}]: {e.Message}");
            }
        }
    }
}
