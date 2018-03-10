namespace Retia.Integration
{
    /// <summary>
    /// An object that can be written to a file.
    /// </summary>
	public interface IFileWritable : IStreamWritable
	{
        /// <summary>
        /// Saves an object to a file.
        /// </summary>
        /// <param name="path">File path to save to.</param>
		void Save(string path);
	}
}