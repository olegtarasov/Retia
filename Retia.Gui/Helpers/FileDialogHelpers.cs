using System;
using Microsoft.Win32;

namespace Retia.Gui.Helpers
{
	public static class FileDialogHelpers
	{
		public static void LoadFiles(Action<string[]> action, string filter, string title = null)
		{
			var dlg = new OpenFileDialog
			{
				Filter = filter,
				Multiselect = true,
                Title = title
			};
			if (dlg.ShowDialog() == true && dlg.FileNames.Length > 0)
			{
				action(dlg.FileNames);
			}
		}

		public static void LoadFile(Action<string> action, string filter, string title = null)
		{
			var dlg = new OpenFileDialog
			{
				Filter = filter,
                Title = title
			};
			if (dlg.ShowDialog() == true && !string.IsNullOrEmpty(dlg.FileName))
			{
				action(dlg.FileName);
			}
		}

		public static void SaveFile(Action<string> action, string filter, string title = null)
		{
			var dlg = new SaveFileDialog
			{
				Filter = filter,
                Title = title
			};
			if (dlg.ShowDialog() == true && !string.IsNullOrEmpty(dlg.FileName))
			{
				action(dlg.FileName);
			}
		}
	}
}