using System.Collections.Generic;
using System.Windows.Controls;
using Retia.Gui.Helpers;
using Retia.Gui.Models;

namespace Retia.Gui.Windows
{
    /// <summary>
    /// Interaction logic for TrainingReportControl.xaml
    /// </summary>
    public partial class TrainingReportControl : UserControl
    {
        private TrainingReportModel Model => (TrainingReportModel)DataContext;

        public TrainingReportControl()
        {
            InitializeComponent();
        }

        private void LastErrorConverter_OnConverting(object sender, ConverterEventArgs e)
        {
            var errors = e.Value as List<double>;
            e.ConvertedValue = errors?[0];
        }

        //private void TMessage_OnTextChanged(object sender, TextChangedEventArgs e)
        //{
        //    tMessage.ScrollToEnd();
        //}
    }
}
