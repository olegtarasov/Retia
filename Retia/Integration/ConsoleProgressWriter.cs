using System;
using System.Text;

namespace Retia.Integration
{
    /// <summary>
    /// Console progress reporter. Prints all messages to stdout.
    /// </summary>
    /// <remarks>
    /// This writer tries its best to preserve output history even when console window is resized. 
    /// But if you play with console size too much and too often, expect some jumbled output.
    /// </remarks>
    public class ConsoleProgressWriter : IProgressWriter
    {
        private readonly bool _alwaysNewLine;

        private int _curItemTop = -1;
        private int _lastOutputTop = -1;
        private int _lastWidth = Console.BufferWidth;
        
        /// <summary>
        /// Default instance of console reporter with alwaysNewLine set to false.
        /// </summary>
        public static readonly ConsoleProgressWriter Instance = new ConsoleProgressWriter();

        /// <summary>
        /// Creates a new console progress reporter.
        /// </summary>
        /// <param name="alwaysNewLine">Whether to print all messages on a new line. Default is false.</param>
        public ConsoleProgressWriter(bool alwaysNewLine = false)
        {
            _alwaysNewLine = alwaysNewLine;
        }

        /// <summary>
        /// Gets a text progress bar useful for monotype display such as a console.
        /// Returns an empty string if progress bar can't occupy at least 12 characters.
        /// </summary>
        /// <param name="value">Current progress.</param>
        /// <param name="maxValue">Max progress.</param>
        /// <param name="otherLen">Status message length. Progress bar will occupy the remaining line space.</param>
        /// <returns>A fancy text progress bar.</returns>
        public static string GetProgressbar(double value, double maxValue, int otherLen)
        {
            int len = Console.BufferWidth - otherLen - 1;
            if (len < Math.Min(Console.BufferWidth - 1, 12))
            {
                return string.Empty;
            }

            int n = (int)Math.Ceiling((value / maxValue) * (len - 2));
            var sb = new StringBuilder();
            sb.Append('|').Append('=', n).Append(' ', len - 2 - n).Append('|');
            return sb.ToString();
        }

        public void SetIntermediate(bool isIntermediate, string message = null)
        {
            Console.WriteLine(message);
            ItemComplete();
        }

        public void SetItemProgress(double value, double maxValue, string message = null)
        {
            var sb = new StringBuilder();
            sb.Append(message).Append(" [").Append(value).Append('/').Append(maxValue).Append(']');
            sb.Append(GetProgressbar(value, maxValue, sb.Length));
            SetItemProgress(sb.ToString());
        }

        public void ItemComplete()
        {
            _curItemTop = -1;
            _lastOutputTop = -1;
        }

        public void SetItemProgress(string message)
        {
            if (_alwaysNewLine)
            {
                Console.WriteLine(message);
                return;
            }

            if (_curItemTop == -1)
            {
                _curItemTop = Console.CursorTop;
                Console.WriteLine(message);
            }
            else
            {
                Console.CursorTop = _curItemTop;
                Console.CursorLeft = 0;
                Console.Write(message);

                int blanks;
                if (_lastWidth != Console.BufferWidth)
                {
                    blanks = (_lastOutputTop - _curItemTop + 1) * _lastWidth;
                    _lastWidth = Console.BufferWidth;
                }
                else
                {
                    blanks = Math.Max((_lastOutputTop - Console.CursorTop) * Console.BufferWidth, 0) + Console.BufferWidth - Console.CursorLeft;
                }

                _lastOutputTop = Console.CursorTop;
                if (blanks > 0)
                {
                    Console.Write(new StringBuilder().Append(' ', blanks).ToString());
                }
                
                Console.CursorLeft = 0;
            }
        }

        public void Message(string message)
        {
            Console.WriteLine(message);
            ItemComplete();
        }
    }
}