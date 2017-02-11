using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Retia.Neural;
using Retia.Neural.Layers;

namespace Retia.Genetic.Neural
{
    public class EvolvableNet : LayeredNet, IEvolvable, IComparable
    {
        private static long maxId = 0;

        public EvolvableNet(int xSize, int hSize, int ySize) 
            : base(1, 1, new GruLayer(xSize, hSize),
                new LinearLayer(hSize, ySize),
                new SoftMaxLayer(ySize))
        {
            Chromosome = new double[Layers.Sum(x => x.TotalParamCount)];
            
            Id = GetId();
        }

        private EvolvableNet(EvolvableNet other) : base(other)
        {
            Fitness = other.Fitness;
            Chromosome = new double[Layers.Sum(x => x.TotalParamCount)];
            other.Chromosome.CopyTo(Chromosome, 0);

            //clones have different IDs!
            Id = GetId();
        }

        public long Id { get; private set; }

        public double Fitness { get; set; }

        public double[] Chromosome { get; private set; }

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
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].ToVectorState(Chromosome, ref idx);
            }
        }

        public void ChromosomeToWeights()
        {
            CheckChromosome();

            int idx = 0;
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].FromVectorState(Chromosome, ref idx);
            }
        }

        private static long GetId()
        {
            maxId++;
            return maxId;
        }

        public override NeuralNet Clone()
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
                Chromosome = new double[Layers.Sum(x => x.TotalParamCount)];
            }
        }
    }
}