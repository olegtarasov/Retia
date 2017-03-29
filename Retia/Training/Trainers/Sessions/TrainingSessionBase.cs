using System;
using System.IO;
using System.Reflection;

namespace Retia.Training.Trainers.Sessions
{
    public abstract class TrainingSessionBase : IDisposable
    {
        protected readonly bool SaveReport;
        protected readonly string SessionDir;

        protected TrainingSessionBase(bool saveReport = true) : this(null, null, saveReport)
        {
        }

        public TrainingSessionBase(string name, bool saveReport = true) : this(name, null, saveReport)
        {
            Name = name;
        }

        public TrainingSessionBase(string name, string baseDirectory, bool saveReport = true)
        {
            SaveReport = saveReport;
            Name = name;

            if (!saveReport)
            {
                return;
            }
            
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
            SessionDir = string.IsNullOrEmpty(baseDir) ? dir : Path.Combine(baseDir, dir);
            Directory.CreateDirectory(SessionDir);
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