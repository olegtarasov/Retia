namespace Retia.Integration
{
    /// <summary>
    /// Typed cloneable interface.
    /// </summary>
    public interface ICloneable<T>
	{
        /// <summary>
        /// Performs a deep clone of an object.
        /// </summary>
        T Clone();
	}
}