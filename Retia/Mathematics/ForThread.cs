using System;
using System.Diagnostics.CodeAnalysis;
using System.Threading;

namespace Retia.Mathematics
{
    public unsafe delegate void ForThreadDelegate(int startIdx, int endIdx, void*[] ptrs);

    public sealed unsafe class ForThread : IDisposable
    {
        private readonly Thread _thread;
        private readonly ManualResetEvent _startEvent;
        private readonly ManualResetEvent _disposeEvent;
        private readonly ManualResetEvent _completeEvent;
        private readonly object _locker = new object();

        private ForThreadDelegate _func;
        private int _startIdx, _endIdx;
        private void*[] _ptrs;

        public WaitHandle CompleteHandle { get; private set; }

        // For testing only.
        [ExcludeFromCodeCoverage]
        internal ForThreadDelegate Func { get { return _func; } set { _func = value; } }

        [ExcludeFromCodeCoverage]
        internal Thread Thread => _thread;

        public ForThread()
        {
            _startEvent = new ManualResetEvent(false);
            _disposeEvent = new ManualResetEvent(false);
            _completeEvent = new ManualResetEvent(false);

            CompleteHandle = _completeEvent;

            _thread = new Thread(Run);
            _thread.Start();
        }

        public void Execute(ForThreadDelegate func, int startIdx, int endIdx, void*[] ptrs)
        {
            lock (_locker)
            {
                _func = func;
                _startIdx = startIdx;
                _endIdx = endIdx;
                _ptrs = ptrs;

                _completeEvent.Reset();
                _startEvent.Set();
            }
        }

        private void Run()
        {
            while (true)
            {
                if (WaitHandle.WaitAny(new WaitHandle[] {_startEvent, _disposeEvent}) == 1)
                {
                    return;
                }

                if (_func == null)
                {
                    continue;
                }

                lock (_locker)
                {
                    _startEvent.Reset();
                    _func(_startIdx, _endIdx, _ptrs);
                    _func = null;
                    _ptrs = null;
                    _completeEvent.Set();
                }
            }
        }

        public void Dispose()
        {
            _disposeEvent.Set();
            if (!_thread.Join(1000))
            {
                _thread.Abort();
            }

            _startEvent.Dispose();
            _disposeEvent.Dispose();
            _completeEvent.Dispose();
        }
    }
}