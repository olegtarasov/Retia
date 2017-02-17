namespace Retia.Mathematics
{
    public class MAV
    {
        private readonly int _order;

        private double _last = 0.0d;
        private double _average = 0.0d;
        private int _ticks = 0;

        public MAV(int order)
        {
            _order = order;
        }

        public double Filter(double value)
        {
            if (_ticks < _order)
            {
                _ticks++;
            }

            if (_ticks == 1)
            {
                _average = value;
            }
            else
            {
                int order = _ticks < _order ? _ticks : _order;
                double cur = value / order;

                if (_ticks < order)
                {
                    _average += cur;
                }
                else
                {
                    _average = _average - _last + cur;
                }
                
                _last = cur;
            }

            return _average;
        }

        public void Reset()
        {
            _average = _last = 0.0d;
            _ticks = 0;
        }
    }
}