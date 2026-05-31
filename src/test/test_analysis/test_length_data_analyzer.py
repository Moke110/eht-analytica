import os
import sys
import traceback

# Ensure the project root directory is on sys.path so we can import functions
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from functions.length_data_analyzer import LengthDataAnalyzer


def pick_csv_folder() -> str:
	"""Open a dialog to select a folder containing length CSV files and return its path (empty string if cancelled)."""
	try:
		import tkinter as tk
		from tkinter import filedialog

		root = tk.Tk()
		root.withdraw()
		root.update()
		folder_path = filedialog.askdirectory(title="Select folder containing {EHT_name}_length.csv files")
		root.destroy()
		return folder_path or ""
	except Exception:
		print("Failed to open folder dialog. Details:\n" + traceback.format_exc())
		return ""


def scan_length_csvs(folder_path: str) -> list[str]:
	"""Return list of *_length.csv files within the selected folder (non-recursive)."""
	if not folder_path or not os.path.isdir(folder_path):
		return []
	files = []
	for name in sorted(os.listdir(folder_path)):
		if name.lower().endswith('_length.csv'):
			files.append(os.path.join(folder_path, name))
	return files

def run_analysis_on_folder(folder_path: str) -> None:
	"""Select all *_length.csv files in a folder, run analysis, and print in-memory summaries."""
	if not folder_path:
		print("No folder selected. Exiting.")
		return
	if not os.path.isdir(folder_path):
		print(f"Folder does not exist: {folder_path}")
		return
	csv_paths = scan_length_csvs(folder_path)
	if not csv_paths:
		print("No *_length.csv files found in folder.")
		return

	analyzer = LengthDataAnalyzer()
	try:
		force_status_dfs, metrics_df = analyzer.analyze_lt_csvs(csv_paths)

		print("Analysis complete (in-memory results):")
		print("  Input folder:", folder_path)
		print("  Discovered files:")
		for src in csv_paths:
			print(f"    - {os.path.basename(src)}")

		print("\n  Force/Status DataFrames:")
		for i, (src, df) in enumerate(zip(csv_paths, force_status_dfs), start=1):
			name = os.path.basename(src)
			nrows = len(df)
			cols = list(df.columns)
			eht_name = df.attrs.get('EHT_name', '')
			print(f"    {i}. {name} (EHT: {eht_name}) -> rows: {nrows}, cols: {cols}")
			# Show a tiny preview
			try:
				print(df.head(3).to_string(index=False))
			except Exception:
				pass
			print("-")

		print("\n  Metrics DataFrame:")
		if metrics_df is not None and not metrics_df.empty:
			print(metrics_df.to_string(index=False))
		else:
			print("    <empty>")
	except Exception as e:
		print("Error during analysis:")
		print(str(e))
		print(traceback.format_exc())


if __name__ == "__main__":
	folder = pick_csv_folder()
	run_analysis_on_folder(folder)

