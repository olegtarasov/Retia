using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using QuickGraph;

using Graph = QuickGraph.BidirectionalGraph<Retia.Analytical.ModelVertex, QuickGraph.Edge<Retia.Analytical.ModelVertex>>;

namespace Retia.Analytical
{
    public class ModelGraph
    {
        public static Graph Create(Dictionary<string, Expression> defs, string[] outputs)
        {
            var vertices = defs.ToDictionary(x => x.Value, x => new ModelVertex(x.Key, x.Value));
            var graph = new Graph();

            foreach (var vertex in vertices.Values)
            {
                graph.AddVertex(vertex);
            }

            foreach (var output in outputs)
            {
                var vert = vertices.Values.FirstOrDefault(x => x.Name == output);
                if (vert == null)
                    throw new InvalidOperationException($"Ouput definition {output} not found!");

                ProcessVertex(vert.Expression, null, vertices, graph);
            }

            return graph;
        }

        private static void ProcessVertex(Expression expr, ModelVertex parent, Dictionary<Expression, ModelVertex> vertices, Graph graph)
        {
            ModelVertex vertex;
            if (!vertices.TryGetValue(expr, out vertex))
            {
                if (expr.NodeType == ExpressionType.Parameter)
                {
                    // This is ugliness, think how to make better.
                    vertex = vertices.Values.FirstOrDefault(x => x.Name == ((ParameterExpression)expr).Name);
                }

                if (vertex == null)
                {
                    vertex = new ModelVertex("", expr);
                    vertices[expr] = vertex;
                    graph.AddVertex(vertex);
                }
            }

            if (expr.NodeType == ExpressionType.Constant)
            {
                if (parent == null)
                    throw new InvalidOperationException("Constant vertex as an output!");

                graph.AddEdge(new Edge<ModelVertex>(vertex, parent));
            }
            else if (expr is BinaryExpression)
            {
                var bin = (BinaryExpression)expr;
                ProcessVertex(bin.Left, vertex, vertices, graph);
                ProcessVertex(bin.Right, vertex, vertices, graph);
            }
            else if (expr.NodeType == ExpressionType.Call)
            {
                var call = (MethodCallExpression)expr;
                if (call.Method.Name == nameof(ModelParser.Gemm))
                {
                    ProcessVertex(call.Arguments[0], vertex, vertices, graph);
                    ProcessVertex(call.Arguments[1], vertex, vertices, graph);
                }
                else if (call.Method.Name == nameof(ModelParser.Func))
                {
                    ProcessVertex(call.Arguments[1], vertex, vertices, graph);
                }
                else
                {
                    throw new InvalidOperationException($"Unsupported method: {call.Method.Name}");
                }

                graph.AddEdge(new Edge<ModelVertex>(vertex, parent));
            }
            else if (expr.NodeType == ExpressionType.Parameter)
            {
                var par = (ParameterExpression)expr;
                var parVert = vertices.Values.FirstOrDefault(x => x.Name == par.Name);
                if (parVert == null)
                    throw new InvalidOperationException($"Undefined parameter: {par.Name}");

                if (parVert.Expression.NodeType == ExpressionType.Parameter)
                {
                    graph.AddEdge(new Edge<ModelVertex>(parVert, parent));
                }
                else
                {
                    ProcessVertex(parVert.Expression, vertex, vertices, graph);
                }
            }
        }
    }
}