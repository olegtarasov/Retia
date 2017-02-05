using System;
using System.IO;
using Retia.Integration;

namespace Retia.Tests.Plumbing
{
    public class DisposableFile : IDisposable
    {
        public readonly string Path;

        public DisposableFile()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.IO.Path.GetRandomFileName());
        }

        public T WriteAndReadData<T>(IStreamWritable data, Func<Stream, T> reader)
        {
            using (var stream = new FileStream(Path, FileMode.Create, FileAccess.Write))
            {
                data.Save(stream);
            }

            using (var stream = new FileStream(Path, FileMode.Open, FileAccess.Read))
            {
                return reader(stream);
            }
        }

        public T WriteAndReadData<T>(Action<Stream> writer, Func<Stream, T> reader)
        {
            using (var stream = new FileStream(Path, FileMode.Create, FileAccess.Write))
            {
                writer(stream);
            }

            using (var stream = new FileStream(Path, FileMode.Open, FileAccess.Read))
            {
                return reader(stream);
            }
        }

        public void Dispose()
        {
            try
            {
                File.Delete(Path);
            }
            catch
            {
            }
        }
    }
}