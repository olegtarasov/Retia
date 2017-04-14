using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using StringToExpression;
using StringToExpression.GrammerDefinitions;
using StringToExpression.Parser;

namespace Retia.Analytical
{
    public class ModelParser
    {
        private static readonly Regex _defRegex = new Regex(@"^\s*(.*?)\s*=\s*(.*)");

        public static decimal Func(string name, decimal p) => 0;
        public static decimal Gemm(decimal a, decimal b) => 0;

        public Dictionary<string, Expression> Parse(string text, string[] pars, string[] inputs)
        {
            var language = CreateLanguage();
            var lines = SplitTrimmedLines(text);
            var defs = new Dictionary<string, Expression>();

            foreach (var par in inputs.Concat(pars))
            {
                defs[par] = Expression.Parameter(typeof(decimal), par);
            }

            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i];

                var match = _defRegex.Match(line);
                if (!match.Success)
                    throw new InvalidOperationException($"Line {i + 1}: Invalid definition format! Must be in form of 'y = ...'");

                string def = match.Groups[1].Value;
                if (string.IsNullOrEmpty(def))
                    throw new InvalidOperationException($"Line {i + 1}: Invalid definition format! Must be in form of 'y = ...'");

                if (defs.ContainsKey(def))
                    throw new InvalidOperationException($"Line {i + 1}: {def} is already defined!");

                defs[def] = language.Parse(match.Groups[2].Value, defs.Keys.Select(x => Expression.Parameter(typeof(decimal), x)).ToArray());
            }

            return defs;
        }

        private Language CreateLanguage()
        {
            ListDelimiterDefinition delimeter;
            BracketOpenDefinition openBracket, func;
            var result = new Language(
                new OperandDefinition(
                    name: "DECIMAL",
                    regex: @"\-?\d+(\.\d+)?",
                    expressionBuilder: x => Expression.Constant(decimal.Parse(x))),
                new BinaryOperatorDefinition(
                    name: "ADD",
                    regex: @"\+",
                    orderOfPrecedence: 2,
                    expressionBuilder: (left, right) => Expression.Add(left, right)),
                new BinaryOperatorDefinition(
                    name: "SUB",
                    regex: @"\-",
                    orderOfPrecedence: 2,
                    expressionBuilder: (left, right) => Expression.Subtract(left, right)),
                new BinaryOperatorDefinition(
                    name: "MUL",
                    regex: @"\*",
                    orderOfPrecedence: 1, //multiply should be done before add/subtract
                    expressionBuilder: (left, right) => Expression.Multiply(left, right)),
                new BinaryOperatorDefinition(
                    name: "DIV",
                    regex: @"\/",
                    orderOfPrecedence: 1, //division should be done before add/subtract
                    expressionBuilder: (left, right) => Expression.Divide(left, right)),
                new BinaryOperatorDefinition(
                    name: "GEMM",
                    regex: @"\^",
                    orderOfPrecedence: 1, //Gemm should be done before add/subtract
                    expressionBuilder: (left, right) =>
                    {
                        return Expression.Call(
                            null,
                            method: typeof(ModelParser).GetMethod("Gemm"),
                            arguments: new[] { left, right });
                    }),
                func = new FunctionCallDefinition(
                    name: "FUNC",
                    regex: @"[a-zA-Z]+\(",
                    argumentTypes: new[] { typeof(decimal) },
                    expressionBuilder: (name, parameters) =>
                    {
                        return Expression.Call(
                            null,
                            method: typeof(ModelParser).GetMethod(nameof(Func)),
                            arguments: new[] { Expression.Constant(name.Substring(0, name.Length - 1)), parameters[0] });
                    }),
                openBracket = new BracketOpenDefinition(
                    name: "OPEN_BRACKET",
                    regex: @"\("),
                delimeter = new ListDelimiterDefinition(
                    name: "COMMA",
                    regex: ","),
                new BracketCloseDefinition(
                    name: "CLOSE_BRACKET",
                    regex: @"\)",
                    bracketOpenDefinitions: new[] {openBracket, func},
                    listDelimeterDefinition: delimeter),
                new OperandDefinition(
                    "SYMBOL",
                    @"[a-zA-Z]+",
                    (s, expressions) =>
                    {
                        var par = expressions.FirstOrDefault(x => x.Name == s);
                        if (par == null)
                            throw new InvalidOperationException($"Undefined parameter {s}!");

                        return par;
                    }),
            new GrammerDefinition(name: "WHITESPACE", regex: @"\s+", ignore: true) //we dont want to process whitespace
                );

            return result;
        }

        private string[] SplitTrimmedLines(string text)
        {
            return text
                .Split(new[] {';'}, StringSplitOptions.RemoveEmptyEntries)
                .Select(x => x.Trim())
                .Where(x => !string.IsNullOrEmpty(x))
                .ToArray();
        }
    }
}
