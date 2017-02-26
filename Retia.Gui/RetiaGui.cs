using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using AmmySidekick;

namespace Retia.Gui
{
    public class RetiaGui
    {
        private readonly Thread _thread;
        
        private Application _application;
        private Func<Window> _windowFunc = null;

        public RetiaGui()
        {
            var dummy = Ammy.RegisterProperty; // We need this for AmmySidekick to get copied to output

            _thread = new Thread(RunApp);
            _thread.SetApartmentState(ApartmentState.STA);
        }

        public void RunAsync(Func<Window> windowFunc)
        {
            _windowFunc = windowFunc;
            _thread.Start();
        }

        public void ShowWindow(Func<Window> windowFunc)
        {
            _application.Dispatcher.Invoke(() => windowFunc().Show());
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