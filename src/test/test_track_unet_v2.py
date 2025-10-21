"""Test script: run EHTTracker.pt on a random subset of test images and
visualize predictions vs ground truth.

Creates PNG visualizations under src/test/test_outputs/.
"""

import os
import sys
import random
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

try:
	import torch
except Exception:
	torch = None


# NOTE: image preprocessing and heatmap -> coords postprocessing are handled
# inside the model's forward() per repo convention. The test code should not
# replicate those steps. We only call the model with the raw image (BGR numpy
# array) and expect coordinates in original image space.


def load_gt(csv_path):
	df = pd.read_csv(csv_path, header=0)
	# Expect two rows, columns x,y (or similar). Try to detect 'x' and 'y'
	cols = [c.lower() for c in df.columns]
	if 'x' in cols and 'y' in cols:
		xcol = df.columns[cols.index('x')]
		ycol = df.columns[cols.index('y')]
		coords = df[[xcol, ycol]].to_numpy()
	else:
		# fallback: take first two columns
		coords = df.iloc[:, :2].to_numpy()
	return coords


def draw_and_save(image_bgr, gt_coords, pred_coords, out_path):
	image_bgr = np.asarray(image_bgr)
	img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	plt.figure(figsize=(6, 6))
	plt.imshow(img_rgb, cmap='gray')
	if gt_coords is not None and len(gt_coords):
		xs = gt_coords[:, 0]
		ys = gt_coords[:, 1]
		plt.scatter(xs, ys, marker='x', c='blue', s=100, linewidths=2, label='GT')
	if pred_coords is not None and len(pred_coords):
		xs = pred_coords[:, 0]
		ys = pred_coords[:, 1]
		plt.scatter(xs, ys, marker='x', c='red', s=100, linewidths=2, label='Pred')
	plt.axis('off')
	plt.legend()
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
	plt.close()


def main():
	repo_root = Path(__file__).resolve().parents[2]

	model_path = repo_root / 'model' / 'track_unet_v2' / 'EHTTracker.pt'
	test_dir = repo_root / 'src' / 'test' / 'track_model_test_set'
	out_dir = repo_root / 'src' / 'test' / 'test_outputs'

	pngs = sorted([p for p in test_dir.glob('*.png')])
	if not pngs:
		print('No PNG test images found in', test_dir)
		return

	# choose 9 random images that have corresponding _coords.csv
	candidates = []
	for p in pngs:
		csv = p.with_name(p.stem + '_coords.csv')
		if csv.exists():
			candidates.append((p, csv))

	if not candidates:
		print('No matching CSV ground-truth files found.')
		return

	selected = random.sample(candidates, min(9, len(candidates)))

	model = None
	if torch is not None and model_path.exists():
		# Load the scripted tracker. We don't import track_unet_v2 here because
		# the TorchScript wrapper does full preprocessing/postprocessing.
		try:
			model = torch.jit.load(str(model_path), map_location='cpu')
			model.eval()
			print('Loaded torchscript model:', model_path)
		except Exception as e:
			print('Could not load torchscript model:', e)
			model = None
	else:
		print('Torch not available or model file missing; will plot GT only.')

	results = []

	for img_path, csv_path in selected:
		img = cv2.imread(str(img_path))
		if img is None:
			print('Could not read image', img_path)
			continue

		gt = load_gt(csv_path)

		pred = None
		if model is not None and torch is not None:
			try:
				# The scripted model expects a grayscale torch.Tensor [H, W]
				arr_img = np.asarray(img)
				if arr_img.ndim == 3:
					gray = cv2.cvtColor(arr_img, cv2.COLOR_BGR2GRAY)
				else:
					gray = arr_img

				tensor = torch.from_numpy(np.asarray(gray)).float()

				with torch.no_grad():
					out = model(tensor)

				if isinstance(out, (list, tuple)):
					out = out[0]

				if isinstance(out, torch.Tensor):
					coords = out.cpu().numpy()
				else:
					coords = np.array(out)

				if coords.ndim == 1 and coords.size >= 4:
					coords = coords.reshape(-1, 2)[:2]

				pred = np.array(coords)
			except Exception as e:
				print('Error calling model on', img_path, '-', e)

		# Build heatmaps and collect results for combined plotting
		pred_heatmap = None
		gt_heatmap = None
		loss_val = np.nan

		if model is not None and torch is not None and pred is not None:
			try:
				# Use exported pre_process and ensemble_forward to get heatmap in 512x512
				pre, orig_w_t, orig_h_t, dx_t, dy_t, S_t = model.pre_process(tensor)
				ensemble = model._ensemble_forward(pre)
				# ensemble: [1,1,512,512]
				pred_heatmap = ensemble[0, 0].detach().cpu().numpy()

				# Create GT heatmap in 512x512
				try:
					S = int(S_t.item())
					dx = float(dx_t.item())
					dy = float(dy_t.item())
				except Exception:
					# fallback if tensors are plain python
					S = int(S_t)
					dx = float(dx_t)
					dy = float(dy_t)

				H = W = 512
				gt_heatmap = np.zeros((H, W), dtype=float)

				if gt is not None and len(gt):
					# Gaussian parameters (match model blob_sd ~=12)
					sd = 12.0
					y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

					for (x_orig, y_orig) in np.asarray(gt).reshape(-1, 2):
						x_canvas = x_orig + dx
						y_canvas = y_orig + dy
						x_512 = x_canvas * (W / float(S))
						y_512 = y_canvas * (H / float(S))
						gauss = np.exp(-((x_grid - x_512) ** 2 + (y_grid - y_512) ** 2) / (2 * sd * sd))
						gt_heatmap += gauss

					# normalize
					maxv = gt_heatmap.max()
					if maxv > 0:
						gt_heatmap = gt_heatmap / maxv

				# normalize predicted heatmap
				ph = pred_heatmap.astype(float)
				ph = ph - ph.min()
				if ph.max() > 0:
					ph = ph / ph.max()
				pred_heatmap = ph

				# compute simple MSE loss between heatmaps
				if gt_heatmap is not None and gt_heatmap.max() > 0:
					loss_val = float(np.mean((pred_heatmap - gt_heatmap) ** 2))
				else:
					loss_val = float(np.mean(pred_heatmap ** 2))

			except Exception as e:
				print('Error building heatmaps for', img_path, '-', e)

		results.append({'img': img, 'gt': gt, 'pred': pred, 'name': img_path.stem,
						'pred_heatmap': pred_heatmap, 'gt_heatmap': gt_heatmap, 'loss': loss_val})

	# After processing all, create the combined summary plot
	if results:
		# compute per-image mean pixel distances (using best assignment between two points)
		def pairwise_mean_distance(gt_pts, pred_pts):
			# gt_pts, pred_pts: (2,2) arrays
			if pred_pts is None or len(pred_pts) == 0:
				return np.nan
			try:
				gt = np.asarray(gt_pts)
				pr = np.asarray(pred_pts)
				if gt.size == 0 or pr.size == 0:
					return np.nan
				# Ensure shapes
				gt = gt.reshape(-1, 2)
				pr = pr.reshape(-1, 2)
				# If counts differ, compute minimal matching for first two
				if gt.shape[0] >= 2 and pr.shape[0] >= 2:
					# two possible assignments: (0->0,1->1) or swapped
					d00 = np.linalg.norm(gt[0] - pr[0]) + np.linalg.norm(gt[1] - pr[1])
					d01 = np.linalg.norm(gt[0] - pr[1]) + np.linalg.norm(gt[1] - pr[0])
					best = min(d00, d01)
					return best / 2.0
				else:
					# fallback: mean distance between available pairs
					m = min(gt.shape[0], pr.shape[0])
					if m == 0:
						return np.nan
					dists_local = [np.linalg.norm(gt[i] - pr[i]) for i in range(m)]
					return float(np.mean(dists_local))
			except Exception:
				return np.nan

		dists = np.array([pairwise_mean_distance(r['gt'], r['pred']) for r in results], dtype=float)
		losses = np.array([r['loss'] if r.get('loss') is not None else np.nan for r in results], dtype=float)
		mean_dist = float(np.nanmean(dists)) if dists.size else float('nan')
		mean_loss = float(np.nanmean(losses)) if losses.size else float('nan')

		# layout: display only original images with GT and Pred crosses; 3 images per row
		per_row = 3
		n = len(results)
		rows = int(np.ceil(n / per_row))
		cols = per_row
		thumb = 256
		fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
		axes = np.array(axes).reshape(rows, cols)

		for idx, r in enumerate(results):
			row = idx // per_row
			col = idx % per_row

			ax = axes[row, col]
			arr_img = np.asarray(r['img'])
			img_rgb = cv2.cvtColor(arr_img, cv2.COLOR_BGR2RGB)
			h, w = img_rgb.shape[:2]
			# downscale preserving original ratio
			max_side = max(h, w)
			if max_side > thumb:
				scale = thumb / float(max_side)
				new_w = max(1, int(w * scale))
				new_h = max(1, int(h * scale))
				try:
					img_disp = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
				except Exception:
					img_disp = img_rgb
			else:
				img_disp = img_rgb

			ax.imshow(img_disp)
			ax.set_title(r['name'])
			# overlay GT and predictions; need to scale markers to displayed size
			if r['gt'] is not None and len(r['gt']):
				gt = np.asarray(r['gt']).reshape(-1, 2)
				xs = gt[:, 0] * (img_disp.shape[1] / float(w))
				ys = gt[:, 1] * (img_disp.shape[0] / float(h))
				ax.scatter(xs, ys, marker='x', c='red', s=80, linewidths=2, label='GT')
			if r['pred'] is not None and len(r['pred']):
				pr = np.asarray(r['pred']).reshape(-1, 2)
				xs = pr[:, 0] * (img_disp.shape[1] / float(w))
				ys = pr[:, 1] * (img_disp.shape[0] / float(h))
				ax.scatter(xs, ys, marker='x', c='blue', s=80, linewidths=2, label='Pred')
			ax.axis('off')
			ax.set_aspect('equal')

		# Hide any unused axes
		for idx in range(n, rows * cols):
			r0 = idx // cols
			c0 = idx % cols
			axes[r0, c0].axis('off')

		# Place summary text centered at bottom
		summary_text = f"Test result of ensemble model on {n} samples: Mean Loss: {mean_loss:.6f} | Mean Pixel Distance: {mean_dist:.2f}"
		fig.text(0.5, 0.02, summary_text, ha='center', fontsize=14)

		# save to script directory named test_{model_name}.png
		script_dir = Path(__file__).parent
		summary_path = script_dir / f'test_{model_path.stem}.png'
		os.makedirs(script_dir, exist_ok=True)
		plt.tight_layout(rect=(0, 0.04, 1, 1))
		fig.savefig(str(summary_path), bbox_inches='tight')
		plt.close(fig)
		print('Saved combined summary plot to', summary_path)


if __name__ == '__main__':
	main()

