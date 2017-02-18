using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace Retia.Gui
{
    public class RetiaGui
    {
        private readonly Thread _thread;

        private Application _application;
        private Func<Window> _windowFunc = null;

        public RetiaGui()
        {
            _thread = new Thread(RunApp);
            _thread.SetApartmentState(ApartmentState.STA);
        }

        public void Run(Func<Window> windowFunc)
        {
            _windowFunc = windowFunc;
            _thread.Start();
        }

        private void RunApp()
        {
            if (_windowFunc == null)
            {
                throw new InvalidOperationException("Set the window!");
            }

            var window = _windowFunc();
            if (window == null)
            {
                throw new InvalidOperationException("Window func returned null!");
            }

            _application = new Application();
            _application.Run(window);
        }
    }
}