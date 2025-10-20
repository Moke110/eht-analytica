"""
Test script for EHTTracker model using 16 random test images.
"""

import os
import glob
import random
import torch
import cv2 as cv
import matplotlib.pyplot as plt


def load_test_sample(image_path):
    """
    Load a test image and its ground truth coordinates.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        image_tensor: Torch tensor [H, W] with pixel values in [0, 255]
        gt_coords: Torch tensor [2] with ground truth coordinates, or None if not available
    """
    # Load image as grayscale
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    
    # Convert to torch tensor [H, W] with float32 dtype
    image_tensor = torch.from_numpy(image).float()
    
    # Try to load corresponding CSV file with ground truth
    csv_path = image_path.replace('.png', '.csv')
    gt_coords = None
    
    if os.path.exists(csv_path):
        try:
            # Read CSV file manually
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Has header and at least one data row
                    # Parse first coordinate pair (skip header)
                    data_line = lines[1].strip()
                    if data_line:
                        parts = data_line.split(',')
                        if len(parts) >= 2:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                gt_coords = torch.tensor([x, y], dtype=torch.float32)
                            except (ValueError, IndexError):
                                pass
        except Exception as e:
            print(f"Warning: Could not load ground truth from {csv_path}: {e}")
    
    return image_tensor, gt_coords


def test_eht_tracker():
    """Test the EHTTracker model with 16 random images from the test set."""
    
    print("="*60)
    print("EHTTracker - Test Script")
    print("="*60)
    
    # Model path
    model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'model', 'track_unet_v2', 'EHTTracker.pt'
    ))
    
    # Load the EHTTracker model
    print(f"\nLoading EHTTracker from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        tracker = torch.jit.load(model_path)
        tracker.eval()
        print("✓ Model loaded successfully!")
        
        # Get file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model file size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get test set directory
    test_set_dir = os.path.join(os.path.dirname(__file__), 'track_model_test_set')
    
    # Find all image files
    image_files = glob.glob(os.path.join(test_set_dir, '*.png'))
    
    if not image_files:
        print(f"Error: No images found in {test_set_dir}")
        return
    
    print(f"Found {len(image_files)} images in test set")
    
    # Select 16 random images
    num_samples = min(16, len(image_files))
    selected_images = random.sample(image_files, num_samples)
    
    print(f"\nTesting on {num_samples} random images:")
    for i, img_path in enumerate(selected_images, 1):
        print(f"  {i}. {os.path.basename(img_path)}")
    
    # Test each image
    results = []
    print("\n" + "="*60)
    print("Running predictions...")
    print("="*60)
    
    for idx, img_path in enumerate(selected_images, 1):
        print(f"\n[{idx}/{num_samples}] Processing: {os.path.basename(img_path)}")
        
        # Load image and ground truth
        image_tensor, gt_coords = load_test_sample(img_path)
        
        if image_tensor is None:
            print(f"  ✗ Failed to load image, skipping...")
            continue
        
        print(f"  Image size: {image_tensor.shape}")
        
        # Predict coordinates using EHTTracker
        try:
            with torch.no_grad():
                pred_coords_all = tracker(image_tensor)  # Returns [num_peaks, 2]
            
            # Use first prediction
            if pred_coords_all.shape[0] > 0:
                pred_coords = pred_coords_all[0]  # [2]
            else:
                pred_coords = torch.zeros(2, dtype=torch.float32)
            
            print(f"  Predicted: ({pred_coords[0].item():.2f}, {pred_coords[1].item():.2f})")
            
            if gt_coords is not None:
                # Calculate error
                error = torch.sqrt(torch.sum((pred_coords - gt_coords) ** 2))
                print(f"  Ground Truth: ({gt_coords[0].item():.2f}, {gt_coords[1].item():.2f})")
                print(f"  Error: {error.item():.2f} pixels")
                
                results.append({
                    'image': os.path.basename(img_path),
                    'pred': pred_coords.cpu().numpy(),
                    'gt': gt_coords.cpu().numpy(),
                    'error': error.item(),
                    'image_tensor': image_tensor
                })
            else:
                print(f"  Ground Truth: Not available")
                results.append({
                    'image': os.path.basename(img_path),
                    'pred': pred_coords.cpu().numpy(),
                    'gt': None,
                    'error': None,
                    'image_tensor': image_tensor
                })
        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate statistics for images with ground truth
    valid_results = [r for r in results if r['error'] is not None]
    
    if valid_results:
        print("\n" + "="*60)
        print("Test Results Summary")
        print("="*60)
        
        errors = [r['error'] for r in valid_results]
        mean_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)
        
        print(f"\nTotal images tested: {num_samples}")
        print(f"Images with ground truth: {len(valid_results)}")
        print(f"\nError Statistics:")
        print(f"  Mean Error:  {mean_error:.2f} pixels")
        print(f"  Max Error:   {max_error:.2f} pixels")
        print(f"  Min Error:   {min_error:.2f} pixels")
        
        # Sort by error for display
        valid_results.sort(key=lambda x: x['error'])
        
        print(f"\nBest predictions:")
        for i, r in enumerate(valid_results[:min(3, len(valid_results))], 1):
            print(f"  {i}. {r['image']}: {r['error']:.2f} pixels")
        
        if len(valid_results) > 3:
            print(f"\nWorst predictions:")
            for i, r in enumerate(valid_results[-3:], 1):
                print(f"  {i}. {r['image']}: {r['error']:.2f} pixels")
    
    # Visualize results
    if results:
        print("\n" + "="*60)
        print("Generating visualization...")
        print("="*60)
        
        num_to_plot = min(16, len(results))
        
        # Create a 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        for i, result in enumerate(results[:num_to_plot]):
            ax = axes[i]
            
            # Display image
            image_np = result['image_tensor'].cpu().numpy()
            ax.imshow(image_np, cmap='gray')
            
            # Plot predicted point
            pred_x, pred_y = result['pred']
            ax.plot(pred_x, pred_y, 'g+', markersize=15, markeredgewidth=2, label='Predicted')
            
            # Plot ground truth if available
            if result['gt'] is not None:
                gt_x, gt_y = result['gt']
                ax.plot(gt_x, gt_y, 'r+', markersize=15, markeredgewidth=2, label='Ground Truth')
                
                # Draw line between pred and gt
                ax.plot([pred_x, gt_x], [pred_y, gt_y], 'y--', linewidth=1, alpha=0.7)
                
                title = f"{result['image']}\nError: {result['error']:.2f}px"
            else:
                title = f"{result['image']}\n(No GT)"
            
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            
            if i == 0:
                ax.legend(loc='upper right', fontsize=6)
        
        # Hide unused subplots
        for i in range(num_to_plot, 16):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(os.path.dirname(__file__), 'test_results_ehttracker.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        
        plt.show()
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    test_eht_tracker()
