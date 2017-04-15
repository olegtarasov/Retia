using GraphSharp.Controls;
using PropertyChanged;
using QuickGraph;
using Retia.Analytical;

namespace Retia.Gui.Graphs
{
    public class GLayout : GraphLayout<Expr, Edge<Expr>, IBidirectionalGraph<Expr, Edge<Expr>>>
    { }

    [ImplementPropertyChanged]
    public class Model
    {
        public IBidirectionalGraph<Expr, Edge<Expr>> Graph { get; set; }
    }
}