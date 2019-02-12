using System.Collections.Generic;
using System.Linq;

namespace Retia.Mathematics
{
    /// <summary>
    /// Simple Moving Average filter.
    /// </summary>
    public class MAV
    {
        private readonly LinkedList<double> _buffer; 

        /// <summary>
        /// Creates a MAV filter with specified order.
        /// </summary>
        /// <param name="order">MAV order.</param>
        public MAV(int order)
        {
            Order = order;
            _buffer = new LinkedList<double>();
        }

        /// <summary>
        /// Gets the order of the filter.
        /// </summary>
        public int Order { get; set; }

        /// <summary>
        /// Returns whether MAV gathered enough data.
        /// </summary>
        public bool IsSaturated => _buffer.Count == Order;

        /// <summary>
        /// Filters the next value.
        /// </summary>
        /// <param name="value">Value to filter.</param>
        /// <returns>Filtered value.</returns>
        public double Filter(double value)
        {
            _buffer.AddLast(value);
            while (_buffer.Count > Order)
            {
                _buffer.RemoveFirst();
            }
            
            return _buffer.Sum() / _buffer.Count;
        }
        
        /// <summary>
        /// Resets the filter.
        /// </summary>
        public void Reset()
        {
            _buffer.Clear();
        }
    }
}