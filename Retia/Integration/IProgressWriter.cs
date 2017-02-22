namespace Retia.Integration
{
    /// <summary>
    /// Unified progress reporting interface. See <see cref="ConsoleProgressWriter"/> for
    /// a console reporter implementation.
    /// </summary>
    public interface IProgressWriter
    {
        /// <summary>
        /// Sets an intermediate state. This makes sense for rich GUI with progress
        /// bars and doesn't make much sense for console.
        /// </summary>
        /// <param name="isIntermediate">Sets whether there is an intermediate state.</param>
        /// <param name="message">Optional message to display.</param>
        void SetIntermediate(bool isIntermediate, string message = null);

        /// <summary>
        /// Sets current item progress. Will not generate a new line on a console
        /// unless <see cref="ItemComplete" /> was called just before this method.
        /// Draws a progress bar on a console.
        /// </summary>
        /// <param name="value">Current progress value.</param>
        /// <param name="maxValue">Maximum progress value.</param>
        /// <param name="message">Optional progress message.</param>
        void SetItemProgress(double value, double maxValue, string message = null);

        /// <summary>
        /// Sets current item progress. Will not generate a new line on a console
        /// unless <see cref="ItemComplete" /> was called just before this method.
        /// Doesn't draw a progress bar on a console.
        /// </summary>
        /// <param name="message"></param>
        void SetItemProgress(string message);

        /// <summary>
        /// Prints a new status message. Always triggers an <see cref="ItemComplete" /> call.
        /// </summary>
        /// <param name="message">Message to print.</param>
        void Message(string message);

        /// <summary>
        /// Instructs the progress writer to display current item completion.
        /// Prints a new line on a console.
        /// </summary>
        void ItemComplete();
    }
}