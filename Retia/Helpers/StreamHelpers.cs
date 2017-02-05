﻿using System;
using System.IO;
using System.Text;
using Retia.Integration;

namespace Retia.Helpers
{
	public static class StreamHelpers
	{
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