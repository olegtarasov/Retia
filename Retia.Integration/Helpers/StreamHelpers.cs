using System;
using System.IO;
using System.Text;

namespace Retia.Integration.Helpers
{
    /// <summary>
    /// Helper functions to deal with streams.
    /// </summary>
	public static class StreamHelpers
	{
        /// <summary>
        /// Creates a binary writer which doesn't close the underlying stream
        /// upon disposal.
        /// </summary>
        /// <param name="stream">Underlying stream.</param>
        public static BinaryWriter NonGreedyWriter(this Stream stream)
	    {
	        return new BinaryWriter(stream, Encoding.UTF8, true);
	    }

        /// <summary>
        /// Creates a binary reader which doesn't close the underlying stream
        /// upon disposal.
        /// </summary>
        /// <param name="stream">Underlying stream.</param>
        public static BinaryReader NonGreedyReader(this Stream stream)
	    {
	        return new BinaryReader(stream, Encoding.UTF8, true);
	    }

        /// <summary>
        /// Loads an object from a path using the supplied function.
        /// </summary>
        /// <typeparam name="T">Object type.</typeparam>
        /// <param name="path">Path to load from.</param>
        /// <param name="loader">Loader function.</param>
        public static T LoadObject<T>(string path, Func<Stream, T> loader)
		{
			if (path == null) throw new ArgumentNullException(nameof(path));
			if (loader == null) throw new ArgumentNullException(nameof(loader));

			using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read))
			{
				return loader(stream);
			}
		}

        /// <summary>
        /// Saves a IStreamWritable object to a specified path.
        /// </summary>
        /// <typeparam name="T">Object type.</typeparam>
        /// <param name="obj">Object to save.</param>
        /// <param name="path">Path to save to.</param>
		public static void SaveObject<T>(this T obj, string path) where T : IStreamWritable
		{
			using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write))
			{
				obj.Save(stream);
			}
		}
	}
}