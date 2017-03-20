namespace Retia.Training.Trainers.Sessions
{
    public class OptimizationError
    {
        public readonly int Iteration;
        public readonly int Epoch;
        public readonly double FilteredError;
        public readonly double RawError;

        public OptimizationError(int iteration, int epoch, double filteredError, double rawError)
        {
            Iteration = iteration;
            Epoch = epoch;
            FilteredError = filteredError;
            RawError = rawError;
        }
    }
}