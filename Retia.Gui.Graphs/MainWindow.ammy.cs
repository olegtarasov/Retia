using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using GraphSharp.Algorithms.Layout.Simple.FDP;
using GraphSharp.Algorithms.Layout.Simple.Hierarchical;
using Retia.Analytical;

using static Retia.Analytical.Expr;

namespace Retia.Gui.Graphs
{
    public partial class MainWindow
    {
        private Expr _model;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_OnLoaded(object sender, RoutedEventArgs e)
        {
            var x = Input("x");
            var hp = Input("hp");

            var r = Sigmoid(Weight("Wxr") * x + Weight("Whr") * hp + Weight("bxr") + Weight("bhr"));
            var z = Sigmoid(Weight("Wxz") * x + Weight("Whz") * hp + Weight("bxz") + Weight("bhz"));
            var hCan = Tanh(Weight("Wxh") * x + (r ^ (Weight("whh") * hp + Weight("bhh"))) + Weight("bxh"));
            var h = ((One() - z) ^ hCan) + (z ^ hp);

            h.Output("y");
            h.State("h");

            _model = h;

            //var test = Sigmoid(Weight("W") * Input("x") + Weight("b")) + Weight("z");
            //test.Output("y");
            //_model = test;

            //graph.LayoutParameters = new EfficientSugiyamaLayoutParameters
            //                         {
            //                             //EdgeRouting = SugiyamaEdgeRoutings.Orthogonal,
            //                             //LayerDistance = 30,
            //                             //VertexDistance = 30
            //                         };
            DataContext = new Model
                          {
                              Graph = _model.Graph
                          };
        }

        private void BtnRelayout_OnClick(object sender, RoutedEventArgs e)
        {
            graph.Relayout();
        }

        private void BtnModel_OnClick(object sender, RoutedEventArgs e)
        {
            DataContext = new Model
                          {
                              Graph = _model.Graph
                          };
        }

        private void BtnDer_OnClick(object sender, RoutedEventArgs e)
        {
            DataContext = new Model
                          {
                              Graph = Analyzer.Stack(_model.Graph)
                          };
        }
    }
}
