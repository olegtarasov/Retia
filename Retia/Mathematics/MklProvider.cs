using System;
using System.IO;
using System.Net;
using System.Reflection;
using Ionic.Zip;
using MathNet.Numerics;
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

            bool complete = false;
            var client = new WebClient();
            client.DownloadProgressChanged += (sender, args) =>
            {
                if (!complete)
                {
                    progressWriter?.SetItemProgress(args.BytesReceived, args.TotalBytesToReceive, "Downloading MKL");
                }
            };

            string path = Path.Combine(Path.GetTempPath(), Path.GetTempFileName() + ".nupkg");
            try
            {
                client.DownloadFileTaskAsync(MKLPackage, path).Wait();
            }
            catch (Exception e)
            {
                progressWriter?.Message($"Failed to download MKL: {e.Message}");
                return false;
            }

            complete = true;
            progressWriter?.ItemComplete();

            try
            {
                string dir = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "x64");
                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                using (var zip = ZipFile.Read(path))
                {
                    var files = zip.SelectEntries("*.*", @"build/x64");
                    foreach (var file in files)
                    {
                        using (var stream = new FileStream(Path.Combine(dir, Path.GetFileName(file.FileName)), FileMode.Create, FileAccess.Write))
                        {
                            file.Extract(stream);
                        }
                    }
                }

                progressWriter?.Message($"Extracted MKL to {dir}");
            }
            catch (Exception e)
            {
                progressWriter?.Message($"Failed to extract MKL: {e.Message}");
                return false;
            }
            finally
            {
                File.Delete(path);
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