using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Retia;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Training.Batching;
using Retia.Training.Data;

namespace LanguageModel
{
	public class TextDataProvider
    {
		private readonly int _batchSize;

		public TextDataProvider(int batchSize)
		{
			_batchSize = batchSize;
		}

		public List<char> Vocab { get; private set; }

        public IDataSet<float> TrainingSet { get; set; }
        public IDataSet<float> TestSet { get; set; }
	    
		public void PrepareTrainingSets(string path, IProgressWriter progressWriter = null)
		{
            Vocab = File.ReadAllText(path).Distinct().OrderBy(x => x).ToList();

			var text = File.ReadAllText(path);

            var batcher = new SequenceBatcher<float, char>(Vocab.Count, (ch, i) => Vocab[i] == ch ? 1 : 0);
			var matrices = batcher.BatchSamples(text.ToList(), new BatchDimension(BatchDimensionType.BatchSize, _batchSize), progressWriter);
			TrainingSet = new SequentialDataSet<float>(matrices);
			TestSet = new SequentialDataSet<float>(new List<Matrix<float>>()); // No testing yet.
		}

		public void LoadVocab(string path)
		{
			using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read))
			using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
			{
				int vocabLen = reader.ReadInt32();
				Vocab = reader.ReadChars(vocabLen).ToList();
			}
		}

		public void Load(string path)
		{
			using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read))
			using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
			{
				int vocabLen = reader.ReadInt32();
				Vocab = reader.ReadChars(vocabLen).ToList();

				TrainingSet = SequentialDataSet<float>.Load(stream);
				TestSet = SequentialDataSet<float>.Load(stream);
			}
		}

		public void Save(string path)
		{
			using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write))
			using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
			{
				writer.Write(Vocab.Count);
				foreach (var ch in Vocab)
				{
					writer.Write(ch);
				}

				TrainingSet.Save(stream);
				TestSet.Save(stream);
			}
		}
	}
}