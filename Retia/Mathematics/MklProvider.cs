using System;
using System.IO;
using System.Net;
using System.Reflection;
using Ionic.Zip;
using MathNet.Numerics;
using Retia.Helpers;
using Retia.Integration;

namespace Retia.Mathematics
{
    /// <summary>
    /// To avoid project directory size explosion due to MKL being copied all over the place, this class was created.
    /// It downloads MKL on demand only when it's needed.
    /// </summary>
    public static class MklProvider
    {
        private const string MKLPackage = "https://www.nuget.org/api/v2/package/MathNet.Numerics.MKL.Win-x64/2.2.0";
        private const string FileName = "MathNet.Numerics.MKL.Win-x64.2.2.0.nupkg";

        /// <summary>
        /// Tries to enable Math.NET native MKL provider. If not found and download is enabled, downloads
        /// and exctracts MKL into [Retia.dll location]\x64.
        /// </summary>
        /// <param name="tryDownload">Whether MKL download is enabled.</param>
        /// <param name="progressWriter">Progress writer.</param>
        /// <returns>True if could use MKL provider, false otherwise.</returns>
        public static bool TryUseMkl(bool tryDownload = true, IProgressWriter progressWriter = null)
        {
            if (Control.TryUseNativeMKL())
            {
                progressWriter?.Message("Using MKL.");
                return true;
            }

            if (!tryDownload)
            {
                progressWriter?.Message("Couldn't use MKL and download is disabled. Using slow math provider.");
                return false;
            }

            progressWriter?.Message("Couldn't use MKL right away, trying to download...");

            string tempPath = Path.Combine(Path.GetTempPath(), FileName);
            string extractDir = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "x64");
            var downloader = new FileDownloader(progressWriter);

            if (!Directory.Exists(extractDir))
            {
                Directory.CreateDirectory(extractDir);
            }
            if (!downloader.DownloadAndExtract(MKLPackage, tempPath, file =>
            {
                var files = file.SelectEntries("*.*", @"build/x64");
                foreach (var entry in files)
                {
                    using (var stream = new FileStream(Path.Combine(extractDir, Path.GetFileName(entry.FileName)), FileMode.Create, FileAccess.Write))
                    {
                        entry.Extract(stream);
                    }
                }
            }, true))
            {
                return false;
            }

            if (!Control.TryUseNativeMKL())
            {
                progressWriter?.Message("Still can't use MKL, giving up.");
                return false;
            }

            return true;
        }
    }
}