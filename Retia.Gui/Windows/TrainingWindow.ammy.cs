using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Retia.Gui.Models;

namespace Retia.Gui.Windows
{
    public partial class TrainingWindow
    {
        public TrainingWindow(TrainingModelBase model)
        {
            InitializeComponent();

            DataContext = model;
        }
    }
}
