using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;

namespace Retia.Codegen
{
    public class Class1
    {
        public void Generate(Document document)
        {
            var tree = document.GetSyntaxTreeAsync().Result;

        }
    }
}
