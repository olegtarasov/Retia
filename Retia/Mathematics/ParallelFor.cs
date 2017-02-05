using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Threading;

namespace Retia.Mathematics
{
    public sealed class ParallelFor : IDisposable
    {
        public static ParallelFor Instance { get; } = new ParallelFor();

        private readonly List<ForThread> _threads;
        private readonly WaitHandle[] _completeHandles;
        private readonly int _procCount;

        // Testing only
        [ExcludeFromCodeCoverage]
        internal List<ForThread> Threads => _threads;

        public ParallelFor() : this(Environment.ProcessorCount)
        {
        }

        internal ParallelFor(int procCount)
        {
            _procCount = procCount;
            _threads = Enumerable.Range(0, _procCount).Select(x => new ForThread()).ToList();
            _completeHandles = _threads.Select(x => x.CompleteHandle).ToArray();
        }

        public unsafe void Execute(ForThreadDelegate func, int count, void*[] ptrs)
        {
            if (func == null) throw new ArgumentNullException(nameof(func));
            if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), "Count should be greater than zero!");

            int batchSize = count / _procCount;
            int rem = count % _procCount;

            int idx;
            int cur = 0;
            for (idx = 0; idx < _procCount; idx++)
            {
                int size = batchSize + (idx < rem ? 1 : 0);
                if (size == 0)
                {
                    break;
                }

                _threads[idx].Execute(func, cur, cur + size, ptrs);
                cur += size;
            }

            WaitHandle[] handles;
            if (idx < _procCount)
            {
                handles = new WaitHandle[idx];
                Array.Copy(_completeHandles, handles, idx);
            }
            else
            {
                handles = _completeHandles;
            }

            WaitHandle.WaitAll(handles);
        }

        public void Dispose()
        {
            for (int i = 0; i < _procCount; i++)
            {
                _threads[i].Dispose();
            }
        }
    }
}