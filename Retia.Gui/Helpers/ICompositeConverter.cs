using System.Windows.Data;

namespace Retia.Gui.Helpers
{
	public interface ICompositeConverter : IValueConverter
	{
		IValueConverter PostConverter { get; set; }
		object PostConverterParameter { get; set; }
	}
}
