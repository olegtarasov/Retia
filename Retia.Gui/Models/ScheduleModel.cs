using System;
using System.Collections.Generic;
using System.Linq;
using PropertyChanged;
using Retia.Training.Trainers.Actions;

namespace Retia.Gui.Models
{
    [ImplementPropertyChanged]
    public class ScheduleModel
    {
        public bool CanChangePeriod { get; set; }
        public PeriodType PeriodType { get; set; }
        public int Period { get; set; }

        public IEnumerable<PeriodType> PeriodTypes => Enum.GetValues(typeof(PeriodType)).Cast<PeriodType>();
    }
}