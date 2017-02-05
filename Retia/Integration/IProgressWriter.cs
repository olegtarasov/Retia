namespace Retia.Integration
{
    public interface IProgressWriter
    {
        void SetIntermediate(bool isIntermediate, string message = null);
        void SetProgress(double value, double maxValue, string message = null);
        void Complete();
    }
}