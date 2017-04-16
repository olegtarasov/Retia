using System;
using System.Collections.Generic;
using System.Linq;
using QuickGraph;
using QuickGraph.Algorithms;

namespace Retia.Analytical
{
    public class Analyzer
    {
        private struct StackedEdge
        {
            public readonly Expr DerTarget;
            public readonly Edge<Expr> Edge;

            public StackedEdge(Expr derTarget, Edge<Expr> edge)
            {
                DerTarget = derTarget;
                Edge = edge;
            }
        }

        public static ExprGraph Stack(ExprGraph graph)
        {
            var result = new ExprGraph();
            var srcToDerMap = new Dictionary<Expr, Expr>();
            var srcSplitMap = new Dictionary<Expr, Expr>();
            var stateMap = new Dictionary<Expr, Expr>();
            var stack = new Stack<StackedEdge>();
            int stateCnt = 0;

            void StackEdges(Expr source, Expr derTarget)
            {
                foreach (var edge in graph.OutEdges(source))
                {
                    stack.Push(new StackedEdge(derTarget, edge));
                }
            }

            (Expr derivative, Expr target) ProcessSourceVertex(Expr source, Edge<Expr> srcEdge)
            {
                Expr split = null;
                Expr der = null;

                if (!srcToDerMap.TryGetValue(source, out der))
                {
                    der = GetDerivative(source);

                    if (der != null)
                    {
                        result.AddVertex(der);
                    }

                    var srcOut = graph.OutEdges(source).ToArray();
                    if (srcOut.Length > 1)
                    {
                        split = ProcessSourceSplit(source);
                        if (der != null)
                        {
                            result.AddEdge(new Edge<Expr>(split, der));
                        }
                        else
                        {
                            der = split;
                        }
                    }

                    if (source.IsMul)
                    {
                        ProcessSourceMul(srcEdge, der);
                    }
                    else
                    {
                        srcToDerMap[source] = split ?? der;
                    }
                }

                return (der, split ?? der);
            }

            Expr ProcessSourceSplit(Expr source)
            {
                if (!srcSplitMap.TryGetValue(source, out var splitSum))
                {
                    splitSum = new Expr(null, ExprType.Add);
                    srcSplitMap[source] = splitSum;
                    result.AddVertex(splitSum);
                }

                return splitSum;
            }

            void ProcessSourceMul(Edge<Expr> srcEdge, Expr der)
            {
                var source = srcEdge.Target;

                if (srcEdge == null)
                    throw new InvalidOperationException("Source multiplication edge is null!");

                var inEdges = graph.InEdges(source).ToArray();
                if (inEdges.Length != 2)
                    throw new InvalidOperationException("Source multiplication has more than 2 in edges!");

                var adj = inEdges.Select(x => x.Source).FirstOrDefault(x => x != srcEdge.Source);
                if (adj == null)
                    throw new InvalidOperationException("Couldn't get adjacent multilication argument!");

                if (!stateMap.TryGetValue(adj, out var state))
                {
                    state = new Expr(adj.Name ?? $"State {stateCnt++}", ExprType.State);
                    stateMap[adj] = state;
                    result.AddVertex(state);
                }

                result.AddEdge(new Edge<Expr>(state, der));
            }

            var roots = graph.Roots().ToArray();

            foreach (var root in roots)
            {
                StackEdges(root, ProcessSourceVertex(root, null).target);
            }

            while (stack.Count > 0)
            {
                var edge = stack.Pop();
                var der = ProcessSourceVertex(edge.Edge.Target, edge.Edge);

                result.AddEdge(new Edge<Expr>(der.derivative, edge.DerTarget));
                if (!srcToDerMap.ContainsKey(edge.Edge.Target))
                    StackEdges(edge.Edge.Target, der.target);
            }

            return result;
        }

        public static ExprGraph Analyze(ExprGraph graph)
        {
            var result = new ExprGraph();

            var srcToDer = new Dictionary<Expr, Expr>();
            var stateCopies = new Dictionary<Expr, Expr>();

            var sinks = graph.Sinks().ToArray();
            if (sinks.Length > 0 && sinks.SelectMany(x => graph.InEdges(x).Select(e => e.Source)).Distinct().Count() > 1)
            {
                throw new InvalidOperationException();
            }

            var queue = new Queue<Expr>();
            foreach (var sink in sinks)
            {
                queue.Enqueue(sink);
            }

            int stateCnt = 0;
            while (queue.Count > 0)
            {
                var src = queue.Dequeue();
                
                // Enqueue all nodes that point to current in the source graph
                foreach (var inVert in graph.InEdges(src).Select(x => x.Source))
                {
                    queue.Enqueue(inVert);
                }

                var der = GetDerivative(src);
                if (der != null)
                {
                    srcToDer[src] = der;
                }

                Expr derEdgeSource = null;

                // Get all the edges that come out of current node in the source
                var srcOut = graph.OutEdges(src).ToArray();
                if (srcOut.Length == 0) // Source is a leaf
                {
                    // Source graph must terminate with State or Output
                    if (der == null)
                        throw new InvalidOperationException("Source graph has Add node as a leaf!");

                    // Just add the vertex and map src to derivative
                    result.AddVertex(der);
                }
                else if (srcOut.Length == 1) // Source node has 1 straight connection to another node
                {
                    if (der != null)
                    {
                        // This is a regular operation, just place it in the graph and connect with prev operation.
                        result.AddVertex(der);
                        derEdgeSource = srcToDer[srcOut[0].Target];
                    }
                    else
                    {
                        // This is summation, which just translates the derivative to subsequent nodes.
                        // We don't need to do anything, just skip the node alltogether.
                        srcToDer[src] = srcToDer[srcOut[0].Target];
                    }
                }
                else 
                {
                    // Source signal is copied to multiple nodes. We must
                    // sum the sensitivities, so we add a new summation node
                    // and connect it to all previous nodes.
                    var sum = new Expr(null, ExprType.Add);
                    result.AddVertex(sum);
                    for (int i = 0; i < srcOut.Length; i++)
                    {
                        result.AddEdge(new Edge<Expr>(srcToDer[srcOut[i].Target], sum));
                    }

                    // If the source operation was not a summation itself,
                    // add the derivative and connect to the new summation node.
                    if (der != null)
                    {
                        result.AddVertex(der);
                        derEdgeSource = sum;
                    }
                    else
                    {
                        // The source node was a summation with immediate split.
                        // We get a single summation in the derivative graph.
                        srcToDer[src] = sum;
                    }
                }

                if (derEdgeSource != null)
                {
                    /*
                     * And now we need to process multiplications.
                     * Since we are going backwards through the source graph,
                     * each source multiplication gives us two derivative
                     * multiplications.
                     */
                    if (derEdgeSource.Type == ExprType.Mul 
                        || derEdgeSource.Type == ExprType.Hadamard)
                    {
                        Expr mulExpr;
                        var inDerSource = result.InEdges(derEdgeSource).ToArray();
                        if (inDerSource.Length > 1)
                        {
                            // We already have one multiplication filled, create another
                            mulExpr = new Expr(null, derEdgeSource.Type);
                            result.AddVertex(mulExpr);

                            var sensSource = inDerSource.Select(x => x.Source).SingleOrDefault(x => x.Type != ExprType.State);
                            if (sensSource == null)
                                throw new InvalidOperationException("Multiplication derivative must have single non-state input!");
                            result.AddEdge(new Edge<Expr>(sensSource, mulExpr));
                        }
                        else
                        {
                            mulExpr = derEdgeSource;
                        }

                        // Now we must find adjacent source node
                        var mulSrc = srcToDer.FirstOrDefault(x => x.Value == derEdgeSource);
                        if (mulSrc.Key == null)
                            throw new InvalidOperationException("Can't find the source for multiplication derivative!");

                        var adj = graph.InEdges(mulSrc.Key).Select(x => x.Source).SingleOrDefault(x => x != src);
                        if (adj == null)
                            throw new InvalidOperationException("Couldn't find adjacent source node for multiplication!");

                        // And connect its copy to our derivative multiplication.
                        if (!stateCopies.TryGetValue(adj, out var stateCopy))
                        {
                            stateCopy = new Expr(adj.Name ?? $"State {stateCnt++}", ExprType.State);
                            stateCopies[adj] = stateCopy;
                            result.AddVertex(stateCopy);
                        }

                        result.AddEdge(new Edge<Expr>(stateCopy, mulExpr));
                        derEdgeSource = mulExpr;
                    }

                    result.AddEdge(new Edge<Expr>(derEdgeSource, der));
                }
            }

            return result;
        }

        private static Expr GetDerivative(Expr source)
        {
            switch (source.Type)
            {
                case ExprType.State:
                    return new Expr(source.Name, ExprType.NextState);
                case ExprType.Output:
                case ExprType.Mul:
                case ExprType.Hadamard:
                    return new Expr(source.Name, source.Type);
                case ExprType.Func:
                    return new Expr(source.Name, ExprType.Derivative);
                case ExprType.Add:
                    return null;
                case ExprType.Input:
                    return new Expr(source.Name, ExprType.InputSens);
                case ExprType.PrevState:
                    return new Expr(source.Name, ExprType.StateSens);
                case ExprType.Weight:
                    return new Expr(source.Name, ExprType.WeightGradient);
            }

            throw new InvalidOperationException("Unsupported expression type!");
        }
    }
}