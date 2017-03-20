using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Retia.Neural;
using Retia.Optimizers;

namespace Retia.Training.Trainers.Sessions
{
    public class OptimizingSession : TrainingSessionBase
    {
        private static readonly List<OptimizationError> FlushInProgress = new List<OptimizationError>();

        private readonly object _flushLock = new object();
        private readonly StreamWriter _errorWriter;
        private readonly string _networksDir;

        private List<OptimizationError> _errorBuffer = new List<OptimizationError>();
        private List<OptimizationError> _swapErrorBuffer = new List<OptimizationError>();
        private bool _flushing = false;


        public OptimizingSession(string name) : this(name, null)
        {
        }

        public OptimizingSession(string name, string baseDirectory) : base(name, baseDirectory)
        {
            _networksDir = Path.Combine(_sessionDir, "Network");
            Directory.CreateDirectory(_networksDir);

            _errorWriter = new StreamWriter(Path.Combine(_sessionDir, "errors.csv"));
            _errorWriter.WriteLine("iterarion;epoch;filtered_error;raw_error");
            _errorWriter.Flush();
        }

        public IOptimizer Optimizer { get; internal set; }
        public INeuralNet Network { get; internal set; }

        public void AddError(double filteredError, double rawError)
        {
            _errorBuffer.Add(new OptimizationError(Iteration, Epoch, filteredError, rawError));
        }

        public List<OptimizationError> GetAndFlushErrors()
        {
            if (_flushing)
                return FlushInProgress;

            lock (_flushLock)
            {
                if (_flushing)
                    return FlushInProgress;

                _flushing = true;
            }

            _swapErrorBuffer = Interlocked.Exchange(ref _errorBuffer, _swapErrorBuffer);

            for (int i = 0; i < _swapErrorBuffer.Count; i++)
            {
                var err = _swapErrorBuffer[i];
                _errorWriter.WriteLine($"{err.Iteration};{err.Epoch};{err.FilteredError};{err.RawError}");
            }

            var result = _swapErrorBuffer.ToList(); // Get a safe copy of errors list.

            _swapErrorBuffer.Clear();
            _errorWriter.Flush();

            _flushing = false;

            return result;
        }

        public void SaveNetwork(int versionsToKeep)
        {
            var files = Directory.GetFiles(_networksDir);
            Array.Sort(files);

            if (files.Length >= versionsToKeep)
            {
                for (int i = 0; i <= files.Length - versionsToKeep; i++)
                {
                    File.Delete(files[i]);
                }
            }

            string fileName = Path.Combine(_networksDir, $"Network_{DateTime.Now:dd.MM.yy_HH.mm.ss}.bin");
            Network.Save(fileName);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            if (disposing)
            {
                GetAndFlushErrors();
                _errorWriter.Dispose();
            }
        }
    }
}