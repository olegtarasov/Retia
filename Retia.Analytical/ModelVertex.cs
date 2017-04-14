using System.Linq.Expressions;

namespace Retia.Analytical
{
    public class ModelVertex
    {
        public ModelVertex(string name, Expression expression)
        {
            Expression = expression;
            Name = name;
        }

        public string Name { get; }
        public Expression Expression { get; }
    }
}