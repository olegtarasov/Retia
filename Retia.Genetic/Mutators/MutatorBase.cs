using System;
using System.Collections.Generic;
using Retia.Genetic.Selectors;

namespace Retia.Genetic.Mutators
{
	public abstract class MutatorBase<T> where T : IEvolvable
	{
	    protected readonly SelectorBase<T> Selector;

        protected MutatorBase(SelectorBase<T> selector)
	    {
	        Selector = selector;
	    }

	    public abstract void Mutate(List<T> population, double maxFitness);
	}
}