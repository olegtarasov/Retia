using System;
using System.Text;

namespace Retia.Training.Trainers
{
    public class ConsoleStatusWriter : ITrainingStatusWriter
    {
        private readonly bool _alwaysNewLine;

        private int _curEpochTop = - 1;

        public ConsoleStatusWriter(bool alwaysNewLine = false)
        {
            _alwaysNewLine = alwaysNewLine;
        }

        public void UpdateEpochStatus(string status)
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
                Console.Write(new StringBuilder().Append(' ', Console.BufferWidth - status.Length).ToString());
                Console.CursorTop = lastTop;
                Console.CursorLeft = lastLeft;
            }
        }

        public void NewLine()
        {
            _curEpochTop = -1;
        }

        public void Message(string message)
        {
            Console.WriteLine(message);
            NewLine();
        }
    }
}