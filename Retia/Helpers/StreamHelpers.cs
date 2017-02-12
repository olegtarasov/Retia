using System;
using System.IO;
using System.Text;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Helpers
{
	public static class StreamHelpers
	{
	    public static T ReadDoubleOrSingle<T>(this BinaryReader reader) where T : struct, IEquatable<T>, IFormattable
	    {
            // This is super-slow, but then again reading from file shouldn't be fast.
	        return (typeof(T) == typeof(double)) 
                ? MathProvider<T>.Instance.Scalar(reader.ReadDouble())
                : MathProvider<T>.Instance.Scalar(reader.ReadSingle());
	    }

	    public static void WriteDoubleOrSingle<T>(this BinaryWriter writer, T value) where T : struct, IEquatable<T>, IFormattable
	    {
            // Heavy boxing for each call
	        if (typeof(T) == typeof(double))
	        {
	            writer.Write(Convert.ToDouble(value));
	        }
	        else
	        {
	            writer.Write(Convert.ToSingle(value));
	        }
	    }


        public static BinaryWriter NonGreedyWriter(this Stream stream)
	    {
	        return new BinaryWriter(stream, Encoding.UTF8, true);
	    }

	    public static BinaryReader NonGreedyReader(this Stream stream)
	    {
	        return new BinaryReader(stream, Encoding.UTF8, true);
	    }

		public static T LoadObject<T>(string path, Func<Stream, T> loader)
		{
			if (path == null) throw new ArgumentNullException(nameof(path));
			if (loader == null) throw new ArgumentNullException(nameof(loader));

			using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read))
			{
				return loader(stream);
			}
		}

		public static void SaveObject<T>(this T obj, string path) where T : IStreamWritable
		{
			using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write))
			{
				obj.Save(stream);
			}
		}
	}
}