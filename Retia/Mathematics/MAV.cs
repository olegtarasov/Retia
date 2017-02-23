using System.Collections.Generic;
using System.Linq;

namespace Retia.Mathematics
{
    /// <summary>
    /// Simple Moving Average filter.
    /// </summary>
    public class MAV
    {
        private readonly int _order;
        private readonly LinkedList<double> _buffer; 

        /// <summary>
        /// Creates a MAV filter with specified order.
        /// </summary>
        /// <param name="order">MAV order.</param>
        public MAV(int order)
        {
            _order = order;
            _buffer = new LinkedList<double>();
        }

        /// <summary>
        /// Filters the next value.
        /// </summary>
        /// <param name="value">Value to filter.</param>
        /// <returns>Filtered value.</returns>
        public double Filter(double value)
        {
            _buffer.AddLast(value);
            if(_buffer.Count > _order)
                _buffer.RemoveFirst();
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