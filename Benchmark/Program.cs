using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CLAP;

namespace Benchmark
{
    internal class Program
    {
        [STAThread]
        internal static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            try
            {
                new Parser<App>().Run(args, new App());
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
