using Microsoft.CodeAnalysis;

foreach (var document in Project.Analysis.Documents)
{
    var root = document.GetSyntaxRootAsync().Result;

    Output.WriteLine($"// {document.FilePath}");
}