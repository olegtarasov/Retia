using System;
using System.IO;
using System.Reflection;
using System.Text;

namespace Retia.Training.Trainers.Sessions
{
    public abstract class TrainingSessionBase : IDisposable
    {
        private static readonly object _addFileLock = new object();
        
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
        public int IterationsPerEpoch { get; set; }
        public int Epoch { get; set; }

        public void AddFileToReport(string fileName, string content)
        {
            using (var stream = new MemoryStream(Encoding.UTF8.GetBytes(content), false))
            {
                AddFileToReport(fileName, stream);
            }
        }

        public void AddFileToReport(string fileName, Stream content)
        {
            if (!SaveReport)
            {
                return;
            }

            lock (_addFileLock)
            {
                string sourcePath = Path.Combine(SessionDir, fileName);
                string dir = Path.GetDirectoryName(sourcePath);
                string name = Path.GetFileNameWithoutExtension(fileName);
                string ext = Path.GetExtension(fileName);
                string fullPath = Path.Combine(dir, $"{name}_e{Epoch}i{Iteration}{ext}");

                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                if (File.Exists(fullPath))
                {
                    var files = Directory.GetFiles(dir, $"{name}*");
                    fullPath = Path.Combine(dir, $"{name}_e{Epoch}i{Iteration}_{files.Length}{ext}");
                }

                using (var stream = new FileStream(fullPath, FileMode.Create, FileAccess.Write))
                {
                    content.CopyTo(stream);
                }
            }
        }

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