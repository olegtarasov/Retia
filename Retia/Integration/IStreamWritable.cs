using System.IO;

namespace Retia.Integration
{
	public interface IStreamWritable
	{
		void Save(Stream stream);
	}
}