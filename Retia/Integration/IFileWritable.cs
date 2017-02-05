namespace Retia.Integration
{
	public interface IFileWritable : IStreamWritable
	{
		void Save(string path);
	}
}