using System;

namespace Retia.Integration
{
    public class ConsoleProgressWriter : IProgressWriter
    {
        public readonly static ConsoleProgressWriter Instance = new ConsoleProgressWriter();

        public void SetIntermediate(bool isIntermediate, string message = null)
        {
            Console.WriteLine(message);
        }

        public void SetProgress(double value, double maxValue, string message = null)
        {
            Console.WriteLine($"{message} {value} of {maxValue}");
        }

        public void Complete()
        {
        }
    }
}