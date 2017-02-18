using System.Windows;
using System.Windows.Controls;
using Retia.Gui.Helpers;
using Retia.Gui.Messages;

namespace Retia.Gui.Windows
{
    /// <summary>
    /// Interaction logic for TrainOptionsControl.xaml
    /// </summary>
    public partial class TrainOptionsControl : UserControl
    {
        public TrainOptionsControl()
        {
            InitializeComponent();
        }

        private void BtnStart_OnClick(object sender, RoutedEventArgs e)
        {
            Post.Box.Publish(Msg.StartTraining, null);
        }

        private void BtnPause_OnClick(object sender, RoutedEventArgs e)
        {
            Post.Box.Publish(Msg.PauseTraining, null);
        }

        private void BtnResume_OnClick(object sender, RoutedEventArgs e)
        {
            Post.Box.Publish(Msg.ResumeTraining, null);
        }

        private void BtnLoadNetwork_OnClick(object sender, RoutedEventArgs e)
        {
            FileDialogHelpers.LoadFile(path => Post.Box.Publish(Msg.LoadNetwork, path), "Bin files (*.bin)|*.bin");
        }
    }
}
