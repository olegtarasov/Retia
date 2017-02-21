using System.Collections.Generic;

namespace Retia.Gui.Helpers
{
    public static class GuiConvert
    {
        public static double? ToLastError(object errors) => (errors as List<double>)?[0];
    }
}