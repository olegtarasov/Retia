using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime;
using System.Text;
using CLAP;
using MathNet.Numerics;
using Retia;

namespace LanguageModel
{
    internal class Program
    {
		private static void Main(string[] args)
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

