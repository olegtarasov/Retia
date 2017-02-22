using System;
using System.Text;
using Retia.Training.Trainers;

namespace Retia.Integration
{
    public class ConsoleStatusWriter : ITrainingStatusWriter
    {
        private readonly bool _alwaysNewLine;

        private int _curEpochTop = - 1;

        public ConsoleStatusWriter(bool alwaysNewLine = false)
        {
            _alwaysNewLine = alwaysNewLine;
        }

        public void UpdateItemStatus(string status)
        {
            if (_alwaysNewLine)
            {
                Console.WriteLine(status);
                return;
            }

            if (_curEpochTop == -1)
            {
                _curEpochTop = Console.CursorTop;
                Console.WriteLine(status);
            }
            else
            {
                int lastTop = Console.CursorTop;
                int lastLeft = Console.CursorLeft;

                Console.CursorTop = _curEpochTop;
                Console.CursorLeft = 0;
                Console.Write(status);
                int blanks = Console.BufferWidth - status.Length;
                if (blanks > 0)
                {
                    Console.Write(new StringBuilder().Append(' ', Console.BufferWidth - status.Length).ToString());
                }
                Console.CursorTop = lastTop;
                Console.CursorLeft = lastLeft;
            }
        }

        public void ItemComplete()
        {
            _curEpochTop = -1;
        }

        public void Message(string message)
        {
            Console.WriteLine(message);
            ItemComplete();
        }
    }
}