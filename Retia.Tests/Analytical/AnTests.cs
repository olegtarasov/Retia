using Retia.Analytical;
using StringToExpression.LanguageDefinitions;
using Xunit;


namespace Retia.Tests.Analytical
{
    public class AnTests
    {
        [Fact]
        public void Test()
        {
            //var expr = Infix.ParseOrThrow("wat(W, x) + b");
            //NewFunction(Expression.Function.)
            //Infix.Format(expr);

            //string wat

            var pars = new[] {"Wxr", "Wxz", "Wxh", "Whr", "Whz", "Whh", "bxr", "bxz", "bxh", "bhr", "bhz", "bhh"};
            var ins = new[] {"x", "hp"};

            string model = @"
                r = sig(Wxr ^ x + Whr ^ hp + bxr + bhr);
                z = sig(Wxz ^ x + Whz ^ hp + bxz + bhz); 
                hCan = tanh(Wxh ^ x + r * (Whh ^ hp + bhh) + bxh);
                h = (1 - z) * hCan + z * hp;";

            var result = new ModelParser().Parse(model, pars, ins);

            var graph = ModelGraph.Create(result, new [] {"h"});
        }
    }
}