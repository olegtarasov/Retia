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
    }
}