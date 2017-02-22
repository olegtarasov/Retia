using System.IO;

namespace Retia.Integration
{
    /// <summary>
    /// Object that can be saved to a stream.
    /// </summary>
	public interface IStreamWritable
	{
        /// <summary>
        /// Saves an object to a stream.
        /// </summary>
        /// <param name="stream">A stream to save to.</param>
		void Save(Stream stream);
	}
}