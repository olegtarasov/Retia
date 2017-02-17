using System.Collections.Generic;
using System.Linq;

namespace Retia.Mathematics
{
    public class MAV
    {
        private readonly int _order;
        private readonly List<double> _buffer; 
        public MAV(int order)
        {
            _order = order;
            _buffer=new List<double>(_order);
        }

        public double Filter(double value)
        {
            _buffer.Add(value);
            if(_buffer.Count>_order)
                _buffer.RemoveAt(0);
            return _buffer.Sum()/_buffer.Count;
        }

        public void Reset()
        {
            _buffer.Clear();
        }
    }
}