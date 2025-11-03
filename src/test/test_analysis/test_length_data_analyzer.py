import os
import sys
import traceback

# Ensure the project src directory is on sys.path so we can import EhtAnalytica
CURRENT_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if SRC_ROOT not in sys.path:
	sys.path.insert(0, SRC_ROOT)

from EhtAnalytica.analyzer.length_data_analyzer import LengthDataAnalyzer


def pick_csv_path() -> str:
	"""Open a file dialog to pick a CSV and return its path (or empty string if cancelled)."""
	try:
		import tkinter as tk
		from tkinter import filedialog

		root = tk.Tk()
		root.withdraw()  # Hide the main window
		root.update()
		csv_path = filedialog.askopenfilename(
			title="Select length-time CSV",
			filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
		)
		root.destroy()
		return csv_path or ""
	except Exception:
		print("Failed to open file dialog. Details:\n" + traceback.format_exc())
		return ""


def run_analysis(csv_path: str) -> None:
	"""Run LengthDataAnalyzer on the provided CSV path and print output locations."""
	if not csv_path:
		print("No file selected. Exiting.")
		return
	if not os.path.isfile(csv_path):
		print(f"File does not exist: {csv_path}")
		return

	analyzer = LengthDataAnalyzer()
	try:
		out_dir = os.path.dirname(csv_path)
		force_identified_paths, metrics_path = analyzer.analyze_lt_csvs([csv_path], output_dir=out_dir)
		print("Analysis complete:")
		for p in force_identified_paths:
			print(f"  Force/Status CSV: {p}")
		print(f"  Metrics CSV:      {metrics_path}")
	except Exception as e:
		print("Error during analysis:")
		print(str(e))
		print(traceback.format_exc())


if __name__ == "__main__":
	path = pick_csv_path()
	run_analysis(path)

