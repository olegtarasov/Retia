using System.Collections.Generic;
using System.Linq;
using GraphSharp.Algorithms.Layout;
using GraphSharp.Controls;
using PropertyChanged;
using QuickGraph;
using Retia.Analytical;

namespace Retia.Gui.Graphs
{
    public class GLayout : GraphLayout<Expr, Edge<Expr>, ExprGraph>
    { }

    [ImplementPropertyChanged]
    public class Model
    {
        public ExprGraph Graph { get; set; }

        public List<string> Layouts => new StandardLayoutAlgorithmFactory<Expr, Edge<Expr>, ExprGraph>().AlgorithmTypes.ToList();

        public string CurLayout { get; set; } = "EfficientSugiyama";
    }
}