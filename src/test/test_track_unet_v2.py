"""
Test script for Track UNet V2 model using real test images.
"""

import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

# Add model directory to path
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'track_unet_v2'))
sys.path.insert(0, model_dir)

from track_unet_v2 import TrackerModel


def load_test_sample(image_path):
    """
    Load an image and its corresponding ground truth coordinates.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (image, ground_truth_coords)
    """
    # Load image
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Load corresponding coordinates
    coords_path = image_path.replace('.png', '_coords.csv')
    coords_df = pd.read_csv(coords_path, header=None)
    
    # Convert to numeric, coercing errors to NaN
    coords_df = coords_df.apply(pd.to_numeric, errors='coerce')
    
    # Drop any rows with NaN values
    coords_df = coords_df.dropna()
    
    coords = coords_df.values.astype(np.float32)
    
    # Sort by x-coordinate
    coords = coords[coords[:, 0].argsort()]
    
    # Only take the first 2 coordinates (in case there are more)
    coords = coords[:2]
    
    return image, coords


def test_tracker_model():
    """Test the tracker model with random images from the test set."""
    
    print("="*60)
    print("Track UNet V2 - Test Script")
    print("="*60)
    
    # Initialize model
    print("\nInitializing TrackerModel...")
    try:
        tracker = TrackerModel(model_dir=model_dir)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nPlease ensure .pth model files are in: {model_dir}")
        return
    
    # Get test set directory
    test_set_dir = os.path.join(os.path.dirname(__file__), 'track_model_test_set')
    
    # Find all image files
    image_files = glob.glob(os.path.join(test_set_dir, '*.png'))
    
    if not image_files:
        print(f"Error: No images found in {test_set_dir}")
        return
    
    print(f"Found {len(image_files)} images in test set")
    
    # Select 4 random images
    num_samples = min(4, len(image_files))
    selected_images = random.sample(image_files, num_samples)
    
    print(f"\nTesting on {num_samples} random images:")
    for img_path in selected_images:
        print(f"  - {os.path.basename(img_path)}")
    
    # Test each image
    results = []
    for img_path in selected_images:
        # Load image and ground truth
        image, gt_coords = load_test_sample(img_path)
        
        # Predict coordinates
        pred_coords = tracker.predict(image, num_peaks=2)
        
        # Calculate error
        errors = np.linalg.norm(pred_coords - gt_coords, axis=1)
        
        results.append({
            'image_path': img_path,
            'image': image,
            'gt_coords': gt_coords,
            'pred_coords': pred_coords,
            'errors': errors
        })
    
    # Calculate overall statistics
    all_errors = np.concatenate([r['errors'] for r in results])
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    max_error = np.max(all_errors)
    min_error = np.min(all_errors)
    
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"Mean error: {mean_error:.2f} ± {std_error:.2f} pixels")
    print(f"Min error:  {min_error:.2f} pixels")
    print(f"Max error:  {max_error:.2f} pixels")
    
    # Print individual results
    print(f"\nIndividual Results:")
    for i, result in enumerate(results):
        img_name = os.path.basename(result['image_path'])
        print(f"\n  Image {i+1}: {img_name}")
        print(f"    Ground Truth: ({result['gt_coords'][0, 0]:.1f}, {result['gt_coords'][0, 1]:.1f}), "
              f"({result['gt_coords'][1, 0]:.1f}, {result['gt_coords'][1, 1]:.1f})")
        print(f"    Predicted:    ({result['pred_coords'][0, 0]:.1f}, {result['pred_coords'][0, 1]:.1f}), "
              f"({result['pred_coords'][1, 0]:.1f}, {result['pred_coords'][1, 1]:.1f})")
        print(f"    Errors:       {result['errors'][0]:.2f} px, {result['errors'][1]:.2f} px")
    
    # Visualize results
    print(f"\n{'='*60}")
    print("Generating visualization...")
    
    fig = plt.figure(figsize=(16, 4 * num_samples))
    
    for i, result in enumerate(results):
        # Image with coordinates
        ax1 = plt.subplot(num_samples, 2, i*2 + 1)
        ax1.imshow(result['image'], cmap='gray')
        ax1.scatter(result['gt_coords'][:, 0], result['gt_coords'][:, 1], 
                   color='red', marker='x', s=200, linewidths=3, label='Ground Truth')
        ax1.scatter(result['pred_coords'][:, 0], result['pred_coords'][:, 1], 
                   color='blue', marker='x', s=200, linewidths=3, label='Predicted')
        
        img_name = os.path.basename(result['image_path'])
        ax1.set_title(f'{img_name}\nMean Error: {np.mean(result["errors"]):.2f} px')
        ax1.legend(loc='upper right')
        ax1.axis('off')
        
        # Error bars
        ax2 = plt.subplot(num_samples, 2, i*2 + 2)
        bar_colors = ['#87CEEB', '#87CEEB']
        ax2.bar(['Circle 1', 'Circle 2'], result['errors'], color=bar_colors, edgecolor='black')
        ax2.axhline(y=np.mean(result['errors']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(result["errors"]):.2f} px')
        ax2.set_ylabel('Error (pixels)')
        ax2.set_title(f'Prediction Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), 'track_model_test_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_tracker_model()
