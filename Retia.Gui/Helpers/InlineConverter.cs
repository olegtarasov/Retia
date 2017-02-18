using System;
using System.Globalization;
using System.Windows.Data;

namespace Retia.Gui.Helpers
{
	public class InlineConverter : IInlineConverter, ICompositeConverter
	{
		public IValueConverter PostConverter { get; set; }
		public object PostConverterParameter { get; set; }
		public event EventHandler<ConverterEventArgs> Converting;
		public event EventHandler<ConverterEventArgs> ConvertingBack;

		public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
		{
			var args = new ConverterEventArgs(value, targetType, parameter, culture);
			var handler = Converting;
            if (handler != null)
			    handler(this, args);
			return PostConverter == null
				? args.ConvertedValue
				: PostConverter.Convert(args.ConvertedValue, targetType, PostConverterParameter, culture);
		}

		public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
		{
			var args = new ConverterEventArgs(value, targetType, parameter, culture);
			var handler = ConvertingBack;
            if (handler != null)
			    handler(this, args);
			return PostConverter == null
				? args.ConvertedValue
				: PostConverter.ConvertBack(args.ConvertedValue, targetType, PostConverterParameter, culture);
		}
	}
}
