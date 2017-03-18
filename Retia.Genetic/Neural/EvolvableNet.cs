using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Retia.Neural;
using Retia.Neural.Layers;

namespace Retia.Genetic.Neural
{
    public class EvolvableNet : LayeredNet<float>, IEvolvable, IComparable
    {
        private static long maxId = 0;

        public EvolvableNet(int xSize, int hSize, int ySize) 
            : base(1, 1, new GruLayer<float>(xSize, hSize),
                new LinearLayer<float>(hSize, ySize),
                new SoftMaxLayer<float>(ySize))
        {
            Chromosome = new float[LayersList.Sum(x => x.TotalParamCount)];
            
            Id = GetId();
        }

        private EvolvableNet(EvolvableNet other) : base(other)
        {
            Fitness = other.Fitness;
            Chromosome = new float[LayersList.Sum(x => x.TotalParamCount)];
            other.Chromosome.CopyTo(Chromosome, 0);

            //clones have different IDs!
            Id = GetId();
        }

        public long Id { get; private set; }

        public double Fitness { get; set; }

        public float[] Chromosome { get; private set; }

        public int CompareTo(object obj)
        {
            var other = obj as EvolvableNet;
            if (Fitness > other.Fitness)
                return -1;
            if (Fitness < other.Fitness)
                return 1;
            return 0;
        }

        public void WeightsToChromosome()
        {
            CheckChromosome();

            int idx = 0;
            for (int i = 0; i < LayersList.Count; i++)
            {
                LayersList[i].ToVectorState(Chromosome, ref idx);
            }
        }

        public void ChromosomeToWeights()
        {
            CheckChromosome();

            int idx = 0;
            for (int i = 0; i < LayersList.Count; i++)
            {
                LayersList[i].FromVectorState(Chromosome, ref idx);
            }
        }

        private static long GetId()
        {
            maxId++;
            return maxId;
        }

        public override NeuralNet<float> Clone()
        {
            return new EvolvableNet(this);
        }

        IEvolvable IEvolvable.Clone()
        {
            return new EvolvableNet(this);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CheckChromosome()
        {
            if (Chromosome == null)
            {
                Chromosome = new float[LayersList.Sum(x => x.TotalParamCount)];
            }
        }
    }
}