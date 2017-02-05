using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Retia.Mathematics;
using Xunit;
using XunitShould;

namespace Retia.Tests.Mathematics
{
    public unsafe class ParallelForTests
    {
        [Fact]
        public void CanRunCalculationInThread()
        {
            var thread = new ForThread();
            bool called = false;
            var func = new ForThreadDelegate((idx, endIdx, ptrs) =>
            {
                idx.ShouldEqual(1);
                endIdx.ShouldEqual(2);
                ptrs.Length.ShouldEqual(1);
                ((int)ptrs[0]).ShouldEqual(42);
                called = true;
            });

            thread.Execute(func, 1, 2, new [] {(void*)new IntPtr(42)});
            thread.CompleteHandle.WaitOne(100);
            called.ShouldBeTrue();
            thread.Dispose();
        }

        [Fact]
        public void CanDisposeThreadWithoutCallingFunc()
        {
            var thread = new ForThread();
            var func = new ForThreadDelegate((idx, endIdx, ptrs) =>
            {
                throw new InvalidOperationException("Func got called, but shouldn't have been.");
            });

            thread.Func = func;
            thread.Dispose();

            thread.Thread.ThreadState.ShouldEqual(ThreadState.Stopped);
        }

        [Fact]
        public void CanAbortLockedThreadOnDispose()
        {
            var thread = new ForThread();
            var evt = new ManualResetEvent(false);
            var func = new ForThreadDelegate((idx, endIdx, ptrs) =>
            {
                evt.WaitOne();
            });

            bool completed = false;
            var waitThread = new Thread(() =>
            {
                thread.CompleteHandle.WaitOne();
                completed = true;
            });
            
            waitThread.Start();
            thread.Execute(func, 0, 0, null);

            thread.Dispose();
            (thread.Thread.ThreadState == ThreadState.AbortRequested || thread.Thread.ThreadState == ThreadState.Aborted).ShouldBeTrue();
            completed.ShouldBeFalse();
            waitThread.Abort();
            evt.Dispose();
        }

        // Ok, this is some convoluted piece of multithreaded testing.
        // The idea is that Execute() blocks until the previous code
        // finished executing.
        [Fact]
        public void CanBlockExecuteUntilPreviousWorkDone()
        {
            var thread = new ForThread();
            var evt1 = new ManualResetEvent(false);
            var evt2 = new ManualResetEvent(false);
            var ctlEvt = new ManualResetEvent(false);
            var firstQueuedEvt = new ManualResetEvent(false);
            var secondQueuedEvt = new ManualResetEvent(false);
            int cnt = 0;
            var func1 = new ForThreadDelegate((idx, endIdx, ptrs) =>
            {
                firstQueuedEvt.Set();
                evt1.WaitOne();
            });
            var func2 = new ForThreadDelegate((idx, endIdx, ptrs) =>
            {
                secondQueuedEvt.Set();
                evt2.WaitOne();
            });

            var waitThread = new Thread(() =>
            {
                thread.CompleteHandle.WaitOne();
                cnt++;
                ctlEvt.Set();
                secondQueuedEvt.WaitOne();
                thread.CompleteHandle.WaitOne();
                cnt++;
                ctlEvt.Set();
            });
            waitThread.Start();

            new Thread(() =>
            {
                thread.Execute(func1, 0, 0, null);
                firstQueuedEvt.WaitOne();
                thread.Execute(func2, 0, 0, null);
            }).Start();

            firstQueuedEvt.WaitOne();
            cnt.ShouldEqual(0);
            evt1.Set();
            ctlEvt.WaitOne();
            cnt.ShouldEqual(1);
            ctlEvt.Reset();
            evt2.Set();
            ctlEvt.WaitOne();
            cnt.ShouldEqual(2);

            thread.Dispose();
        }

        [Fact]
        public void WillDoNothingWhenFunctionIsNotSet()
        {
            var thread = new ForThread();

            thread.Execute(null, 0, 0, null);
        }

        private static IEnumerable<object[]> GetParallelForTestData()
        {
            yield return new object[] {4, 4, 4};
            yield return new object[] { 4, 3, 3 };
            yield return new object[] { 4, 1, 1 };
            yield return new object[] { 4, 8, 4 };
            yield return new object[] { 4, 9, 4 };
            yield return new object[] { 4, 10, 4 };
            yield return new object[] { 4, 11, 4 };
            yield return new object[] { 4, 12, 4 };
            yield return new object[] { 1, 4, 1 };
            yield return new object[] { 1, 1, 1 };
        }

        [Theory]
        [MemberData(nameof(GetParallelForTestData))]
        public void CanExecuteInParallel(int procCount, int itemCount, int actualRunCount)
        {
            var flags = new bool[itemCount];
            int runCnt = 0;
            var func = new ForThreadDelegate((idx, endIdx, ptrs) =>
            {
                Interlocked.Increment(ref runCnt);
                for (int i = idx; i < endIdx; i++)
                {
                    flags[i] = true;
                }
            });

            var pf = new ParallelFor(procCount);
            pf.Execute(func, itemCount, null);

            flags.ShouldNotContain(false);
            runCnt.ShouldEqual(actualRunCount);
        }

        [Fact]
        public void CantExecuteWhenFuncNotSet()
        {
            Trap.Exception(() => ParallelFor.Instance.Execute(null, 1, null)).ShouldBeInstanceOf<ArgumentNullException>();
        }

        [Fact]
        public void CantExecuteWhenCountIsZeroOrLess()
        {
            Trap.Exception(() => ParallelFor.Instance.Execute((idx, endIdx, ptrs) => { }, 0, null)).ShouldBeInstanceOf<ArgumentOutOfRangeException>();
            Trap.Exception(() => ParallelFor.Instance.Execute((idx, endIdx, ptrs) => { }, -1, null)).ShouldBeInstanceOf<ArgumentOutOfRangeException>();
        }

        [Fact]
        public void CanDisposeThreadPool()
        {
            var pf = new ParallelFor(4);

            var threads = pf.Threads;
            foreach (var thread in threads)
            {
                thread.Thread.ThreadState.ShouldEqual(ThreadState.WaitSleepJoin);
            }

            pf.Dispose();

            foreach (var thread in threads)
            {
                var state = thread.Thread.ThreadState;
                state.ShouldEqual(ThreadState.Stopped);
            }
        }
    }
}