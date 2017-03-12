using System;
using System.IO;
using System.Net;
using System.Reflection;
using Ionic.Zip;
using Retia.Integration;

namespace Retia.Helpers
{
    /// <summary>
    /// This class downloads and extracts zip archives.
    /// </summary>
    public class FileDownloader
    {
        private readonly IProgressWriter _progressWriter;

        /// <summary>
        /// Creates a new file downloader.
        /// </summary>
        /// <param name="progressWriter">Optional progress writer.</param>
        public FileDownloader(IProgressWriter progressWriter = null)
        {
            _progressWriter = progressWriter;
        }

        /// <summary>
        /// Downloads a file to <see cref="downloadFileName"/> and extracts contents using <see cref="extractor"/>.
        /// </summary>
        /// <param name="url">Url to download the file from.</param>
        /// <param name="downloadFileName">Path to download the file to.</param>
        /// <param name="extractor">Action to extract file with.</param>
        /// <param name="keepDownloaded">Keep downloaded file. If false, file is removed after exctraction.</param>
        /// <returns>True if suceeded, false otherwise.</returns>
        public bool DownloadAndExtract(string url, string downloadFileName, Action<ZipFile> extractor, bool keepDownloaded = false)
        {
            bool complete = false;
            var client = new WebClient();
            client.DownloadProgressChanged += (sender, args) =>
            {
                if (!complete)
                {
                    _progressWriter?.SetItemProgress(args.BytesReceived, args.TotalBytesToReceive, "Downloading MKL");
                }
            };

            try
            {
                client.DownloadFileTaskAsync(url, downloadFileName).Wait();
            }
            catch (Exception e)
            {
                _progressWriter?.Message($"Failed to download MKL: {e.Message}");
                return false;
            }

            complete = true;
            _progressWriter?.ItemComplete();

            try
            {
                using (var zip = ZipFile.Read(downloadFileName))
                {
                    extractor(zip);
                }

                _progressWriter?.Message($"Extracted MKL to {downloadFileName}");
            }
            catch (Exception e)
            {
                _progressWriter?.Message($"Failed to extract MKL: {e.Message}");
                return false;
            }
            finally
            {
                if (!keepDownloaded)
                {
                    File.Delete(downloadFileName);
                }
            }

            return true;
        }
    }
}