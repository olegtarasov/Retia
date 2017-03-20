using System;
using System.IO;
using System.Reflection;

namespace Retia.Training.Trainers.Sessions
{
    public abstract class TrainingSessionBase : IDisposable
    {
        protected readonly string _sessionDir;

        public TrainingSessionBase(string name) : this(name, null)
        {
            Name = name;
        }

        public TrainingSessionBase(string name, string baseDirectory)
        {
            Name = name;

            string baseDir = baseDirectory;
            if (string.IsNullOrEmpty(baseDir))
            {
                baseDir = Environment.CurrentDirectory;
            }

            if (string.IsNullOrEmpty(baseDir))
            {
                baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            }

            string dir = $"{(string.IsNullOrEmpty(Name) ? "Unnamed" : Name)}_{DateTime.Now:dd.MM.yy_HH.mm.ss}";
            _sessionDir = string.IsNullOrEmpty(baseDir) ? dir : Path.Combine(baseDir, dir);
            Directory.CreateDirectory(_sessionDir);
        }

        public string Name { get; private set; }
        public int Iteration { get; set; }
        public int Epoch { get; set; }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}