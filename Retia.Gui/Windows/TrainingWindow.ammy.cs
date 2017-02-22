using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using Retia.Gui.Models;

namespace Retia.Gui.Windows
{
    public partial class TrainingWindow : Window
    {
        public TrainingWindow(TrainingModelBase model)
        {
            InitializeComponent();

            DataContext = model;
        }
    }
}
