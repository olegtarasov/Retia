using System;
using System.Linq;
using QuickGraph;
using QuickGraph.Algorithms;
using Graph = Retia.Analytical.ExprGraph;

namespace Retia.Analytical
{
    public enum ExprType
    {
        Input,
        Output,
        Weight,
        State,
        PrevState,
        Add,
        Sub,
        Mul,
        Hadamard,
        Func,
        One,

        // Derivatives
        Derivative,
        InputSens,
        WeightGradient,
        StateSens,
        NextState
    }

    public class Expr
    {
        private readonly Graph _graph;

        public Expr(string name, ExprType type)
        {
            Name = name;
            Type = type;

            _graph = new Graph();
            _graph.AddVertex(this);
        }

        private Expr(Expr left, Expr right, ExprType type)
        {
            Type = type;
            _graph = ConcatGraphs(left._graph, right._graph);
        }


        private Expr(string name, Expr operand, ExprType type)
        {
            Name = name;
            Type = type;
            _graph = ConcatGraphs(operand._graph);
        }

        public string Name { get; set; }
        public ExprType Type { get; set; }
        public ExprGraph Graph => _graph;

        public void Output(string name)
        {
            var output = new Expr(name, ExprType.Output);
            _graph.AddVertex(output);
            _graph.AddEdge(new Edge<Expr>(this, output));
        }

        public void State(string name)
        {
            var state = new Expr(name, ExprType.State);
            _graph.AddVertex(state);
            _graph.AddEdge(new Edge<Expr>(this, state));
        }

        public static Expr Input(string name)
        {
            return new Expr(name, ExprType.Input);
        }

        public static Expr Weight(string name)
        {
            return new Expr(name, ExprType.Weight);
        }

        public static Expr operator +(Expr left, Expr right)
        {
            return new Expr(left, right, ExprType.Add);
        }

        public static Expr operator -(Expr left, Expr right)
        {
            if (left.Type != ExprType.One)
                throw new InvalidOperationException("Subtraction is allowed only for 1 - x");

            return Func("OneSub", right);
        }

        public static Expr operator *(Expr left, Expr right)
        {
            return new Expr(left, right, ExprType.Mul);
        }

        public static Expr operator ^(Expr left, Expr right)
        {
            return new Expr(left, right, ExprType.Hadamard);
        }

        public static Expr Func(string name, Expr operand)
        {
            return new Expr(name, operand, ExprType.Func);
        }

        public static Expr Sigmoid(Expr operand)
        {
            return Func("sig", operand);
        }

        public static Expr Tanh(Expr operand)
        {
            return Func("tanh", operand);
        }

        public static Expr One()
        {
            return new Expr(null, ExprType.One);
        }

        public static Expr PrevState(string name)
        {
            return new Expr(name, ExprType.PrevState);
        }

        private Graph ConcatGraphs(params Graph[] graphs)
        {
            var result = new Graph();
            for (int i = 0; i < graphs.Length; i++)
            {
                result.AddVertexRange(graphs[i].Vertices);
                result.AddEdgeRange(graphs[i].Edges);
            }
            result.AddVertex(this);

            foreach (var sink in graphs.SelectMany(x => x.Sinks()))
            {
                result.AddEdge(new Edge<Expr>(sink, this));
            }

            return result;
        }

        public override string ToString()
        {
            return $"[{Type}]{Name}";
        }
    }
}