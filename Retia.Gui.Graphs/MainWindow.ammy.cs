using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using Retia.Analytical;

using static Retia.Analytical.Expr;

namespace Retia.Gui.Graphs
{
    public partial class MainWindow
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_OnLoaded(object sender, RoutedEventArgs e)
        {
            //var pars = new[] { "Wxr", "Wxz", "Wxh", "Whr", "Whz", "Whh", "bxr", "bxz", "bxh", "bhr", "bhz", "bhh" };
            //var ins = new[] { "x", "hp" };

            //string model = @"
            //    r = sig(Wxr * x + Whr * hp + bxr + bhr);
            //    z = sig(Wxz * x + Whz * hp + bxz + bhz); 
            //    hCan = tanh(Wxh * x + r ^ (Whh * hp + bhh) + bxh);
            //    h = (1 - z) ^ hCan + z ^ hp;";

            //var result = new ModelParser().Parse(model, pars, ins);

            //var graph = CalculationGraph.Create(result, new[] { "h" });

            var x = Input("x");
            var hp = Input("hp");

            var r = Sigmoid(Weight("Wxr") * x + Weight("Whr") * hp + Weight("bxr") + Weight("bhr"));
            var z = Sigmoid(Weight("Wxz") * x + Weight("Whz") * hp + Weight("bxz") + Weight("bhz"));
            var hCan = Tanh(Weight("Wxh") * x + (r ^ (Weight("whh") * hp + Weight("bhh"))) + Weight("bxh"));
            var h = ((One() - z) ^ hCan) + (z ^ hp);

            h.Output("y");
            h.State("h");

            var lin = Weight("W") * Input("x") + Weight("b");
            lin.Output("y");

            DataContext = new Model
                          {
                              //Graph = der
                              Graph = Analyzer.Stack(lin.Graph)
                          };
        }
    }
}
