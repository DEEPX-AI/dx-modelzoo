#!/usr/bin/env python3
"""
Model Comparison Tool for ONNX vs DXNN Analysis

This tool compares inference results between ONNX Runtime and DX Runtime,
providing detailed similarity metrics and visualizations.

Usage:
    python -m dx_modelzoo.tools.model_comparison \\
        --model yolov5 \\
        --onnx models/yolov5.onnx \\
        --dxnn models/yolov5.dxnn \\
        --image test.jpg \\
        --output comparison.png

Architecture aligned with dx-modelzoo project patterns.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
from dataclasses import dataclass
from scipy.spatial.distance import cosine

# Import from dx-modelzoo
from dx_modelzoo.enums import SessionType
from dx_modelzoo.models import ModelBase
from dx_modelzoo.factory.model_factory import ModelFactory
from dx_modelzoo.session import SessionBase


@dataclass
class ComparisonMetrics:
    """Metrics for comparing ONNX vs DXNN outputs"""
    cosine_similarity: float
    pearson_correlation: float
    mse: float
    mae: float
    normalized_rmse: float
    ssim: float
    relative_error: float
    
    def print(self) -> None:
        """Print metrics in formatted way"""
        # Calculate overall quality assessment (same as original code)
        avg_similarity = np.mean([self.cosine_similarity, self.pearson_correlation, self.ssim])
        
        if avg_similarity > 0.99:
            quality = "EXCELLENT"
            quality_desc = "Highly Similar (>99%)"
        elif avg_similarity > 0.95:
            quality = "GOOD"
            quality_desc = "Quite Similar (>95%)"
        elif avg_similarity > 0.90:
            quality = "ACCEPTABLE"
            quality_desc = "Reasonably Similar (>90%)"
        else:
            quality = "POOR"
            quality_desc = "Significant Differences (<90%)"
        
        print("\n" + "="*60)
        print("Similarity Analysis Results (ONNX vs DXNN)")
        print("="*60)
        print(f"  Cosine Similarity:            {self.cosine_similarity:.6f}  (1.0 = identical)")
        print(f"  Pearson Correlation:          {self.pearson_correlation:.6f}  (1.0 = identical)")
        print(f"  Mean Squared Error (MSE):     {self.mse:.6f}  (lower is better)")
        print(f"  Mean Absolute Error (MAE):    {self.mae:.6f}  (lower is better)")
        print(f"  Normalized RMSE:              {self.normalized_rmse:.6f}  (0.0 = identical)")
        print(f"  SSIM:                         {self.ssim:.6f}  (1.0 = identical)")
        print(f"  Relative Error:               {self.relative_error:.2f}%  (0% = identical)")
        print("="*60)
        print(f"  Overall Assessment: {quality} - {quality_desc}")
        print("="*60 + "\n")


class ModelComparator:
    """
    Compare ONNX and DXNN model outputs
    
    This class handles:
    - Loading models via ModelFactory
    - Running inference on both runtimes
    - Computing similarity metrics
    - Generating comparison visualizations
    """
    
    def __init__(
        self,
        model_name: str,
        onnx_path: str,
        dxnn_path: str,
        data_dir: str = ".",
        image_index: int = 0
    ):
        """
        Initialize comparator
        
        Args:
            model_name: Name of model (e.g., 'yolov5', 'resnet50')
            onnx_path: Path to ONNX model file
            dxnn_path: Path to DXNN model file
            data_dir: Directory for dataset (required for evaluator)
            image_index: Index of image to use from dataset (default: 0)
        """
        self.model_name = model_name
        self.onnx_path = Path(onnx_path)
        self.dxnn_path = Path(dxnn_path)
        self.data_dir = data_dir
        self.image_index = image_index
        
        # Validate paths
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not self.dxnn_path.exists():
            raise FileNotFoundError(f"DXNN model not found: {dxnn_path}")
        
        # Create model factories
        print(f"\n{'='*60}")
        print(f"Initializing Model Comparator")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"ONNX: {self.onnx_path}")
        print(f"DXNN: {self.dxnn_path}")
        print(f"Data Dir: {data_dir}")
        print(f"Image Index: {image_index}")
        print(f"{'='*60}\n")
        
        # Create ONNX factory and model
        self.onnx_factory = ModelFactory(
            model_name=model_name,
            session_type=SessionType.onnxruntime,
            model_path=str(self.onnx_path),
            data_dir=data_dir
        )
        self.onnx_model: ModelBase = self.onnx_factory.make_model()
        
        # Create DXNN factory and model
        self.dxnn_factory = ModelFactory(
            model_name=model_name,
            session_type=SessionType.dxruntime,
            model_path=str(self.dxnn_path),
            data_dir=data_dir
        )
        self.dxnn_model: ModelBase = self.dxnn_factory.make_model()
        
        # Get sessions and datasets from evaluators
        self.onnx_session: SessionBase = self.onnx_model.evaluator.session
        self.dxnn_session: SessionBase = self.dxnn_model.evaluator.session
        self.dataset = self.onnx_model.evaluator.dataset  # Both use same dataset
        
        # Get sample image from dataset
        print(f"\nLoading image from dataset (index: {image_index})...")
        # Dataset returns (preprocessed_img, original_shape, image_id)
        # We need to load the original image directly
        if hasattr(self.dataset, 'img_files'):
            # For datasets with img_files attribute (COCO, ImageNet, etc.)
            image_path = self.dataset.img_files[image_index]
            self.original_image = cv2.imread(image_path)
            print(f"  Image path: {image_path}")
        else:
            # Fallback: use preprocessed image from dataset
            self.original_image = self.dataset[image_index][0]
            print(f"  Warning: Using preprocessed image from dataset")
        
        print(f"  Image shape: {self.original_image.shape}")
        print(f"  Image dtype: {self.original_image.dtype}")
        
        # Print model info
        print("\nModel Information:")
        self.onnx_model.info.print()
        
    def run_inference(
        self,
        session_type: SessionType
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Run inference using dataset's preprocessing
        
        Args:
            session_type: ONNXRUNTIME or DXRUNTIME
            
        Returns:
            Tuple of (preprocessed_input, raw_output, postprocessed_output)
        """
        if session_type == SessionType.onnxruntime:
            model = self.onnx_model
            session = self.onnx_session
            evaluator = self.onnx_model.evaluator
            dataset = evaluator.dataset
        else:
            model = self.dxnn_model
            session = self.dxnn_session
            evaluator = self.dxnn_model.evaluator
            dataset = evaluator.dataset
        
        # Use dataset's preprocessing (Compose transforms set by model)
        # Dataset preprocessing is already set by model's set_preprocessing()
        preprocessed = dataset.preprocessing(self.original_image)
        
        # Convert to torch tensor if it's numpy array
        if isinstance(preprocessed, np.ndarray):
            preprocessed = torch.from_numpy(preprocessed)
        
        # Add batch dimension if needed
        if len(preprocessed.shape) == 3:
            preprocessed = preprocessed.unsqueeze(0)
        
        print(f"\n{session_type.value} Preprocessing:")
        print(f"  Input shape: {self.original_image.shape}")
        print(f"  Preprocessed shape: {preprocessed.shape}")
        print(f"  Preprocessed dtype: {preprocessed.dtype}")
        print(f"  Value range: [{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
        
        # Run inference
        raw_output = session.run(preprocessed)
        
        # Handle different output formats
        if isinstance(raw_output, (list, tuple)):
            raw_output = raw_output[0]
        
        print(f"  Output shape: {raw_output.shape}")
        print(f"  Output dtype: {raw_output.dtype}")
        print(f"  Output range: [{raw_output.min():.4f}, {raw_output.max():.4f}]")
        
        # Postprocess - use evaluator's postprocessing
        postprocessed = evaluator.postprocessing(raw_output)
        
        return preprocessed, raw_output, postprocessed
    
    def calculate_metrics(
        self,
        onnx_output: np.ndarray,
        dxnn_output: np.ndarray
    ) -> ComparisonMetrics:
        """
        Calculate similarity metrics between outputs (same as original yolo_analysis.py)
        
        Args:
            onnx_output: ONNX model output
            dxnn_output: DXNN model output
            
        Returns:
            ComparisonMetrics dataclass
        """
        # Ensure same shape
        if onnx_output.shape != dxnn_output.shape:
            print(f"\nWarning: Output shapes differ!")
            print(f"  ONNX: {onnx_output.shape}")
            print(f"  DXNN: {dxnn_output.shape}")
        
        # Flatten for metrics
        flat1 = onnx_output.flatten()
        flat2 = dxnn_output.flatten()
        
        # Align lengths if needed
        min_len = min(len(flat1), len(flat2))
        flat1 = flat1[:min_len]
        flat2 = flat2[:min_len]
        
        # 1. Cosine Similarity
        cosine_similarity = 1 - cosine(flat1, flat2)
        
        # 2. Pearson Correlation
        try:
            pearson_correlation = np.corrcoef(flat1, flat2)[0, 1]
        except:
            pearson_correlation = 0.0
        
        # 3. Mean Squared Error (MSE)
        mse = np.mean((flat1 - flat2) ** 2)
        
        # 4. Mean Absolute Error (MAE)
        mae = np.mean(np.abs(flat1 - flat2))
        
        # 5. Normalized Root Mean Squared Error (NRMSE)
        rmse = np.sqrt(mse)
        data_range = np.max(flat1) - np.min(flat1)
        normalized_rmse = rmse / data_range if data_range != 0 else 0
        
        # 6. Structural Similarity Index (SSIM-like)
        mean1, mean2 = np.mean(flat1), np.mean(flat2)
        var1, var2 = np.var(flat1), np.var(flat2)
        covar = np.mean((flat1 - mean1) * (flat2 - mean2))
        
        c1, c2 = 0.01, 0.03
        ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
               ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
        
        # 7. Relative Error (percentage, only for non-zero values)
        # Calculate relative error only where |flat1| > threshold to avoid division issues
        threshold = np.abs(flat1).max() * 0.01  # 1% of max value
        valid_mask = np.abs(flat1) > threshold
        if valid_mask.sum() > 0:
            relative_error = np.mean(np.abs(flat1[valid_mask] - flat2[valid_mask]) / np.abs(flat1[valid_mask])) * 100
        else:
            relative_error = 0.0
        
        return ComparisonMetrics(
            cosine_similarity=float(cosine_similarity),
            pearson_correlation=float(pearson_correlation),
            mse=float(mse),
            mae=float(mae),
            normalized_rmse=float(normalized_rmse),
            ssim=float(ssim),
            relative_error=float(relative_error)
        )
    
    def visualize_comparison(
        self,
        original_image: np.ndarray,
        onnx_output: Dict[str, Any],
        dxnn_output: Dict[str, Any],
        metrics: ComparisonMetrics,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization comparing ONNX vs DXNN results
        
        Args:
            original_image: Original input image
            onnx_output: ONNX postprocessed output
            dxnn_output: DXNN postprocessed output
            metrics: Comparison metrics
            output_path: Path to save visualization (optional)
            
        Returns:
            Visualization image
        """
        # Create side-by-side comparison
        h, w = original_image.shape[:2]
        
        # Create canvas
        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Original image (top-left)
        canvas[:h, :w] = original_image
        
        # ONNX result (top-right)
        onnx_vis = self._draw_predictions(original_image.copy(), onnx_output, "ONNX")
        canvas[:h, w:] = onnx_vis
        
        # DXNN result (bottom-left)
        dxnn_vis = self._draw_predictions(original_image.copy(), dxnn_output, "DXNN")
        canvas[h:, :w] = dxnn_vis
        
        # Metrics (bottom-right)
        metrics_vis = self._draw_metrics(w, h, metrics)
        canvas[h:, w:] = metrics_vis
        
        # Add labels
        cv2.putText(canvas, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "ONNX Runtime", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(canvas, "DX Runtime (NPU)", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(canvas, "Metrics", (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, canvas)
            print(f"\nVisualization saved to: {output_path}")
        
        return canvas
    
    def _draw_predictions(
        self,
        image: np.ndarray,
        predictions: Dict[str, Any],
        label: str
    ) -> np.ndarray:
        """Draw predictions on image (for object detection models)"""
        # For classification models
        if 'top1_class' in predictions:
            text = f"{label}: Class {predictions['top1_class']} ({predictions['top1_score']:.3f})"
            cv2.putText(image, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show top-5
            y_offset = 90
            for i, (cls, score) in enumerate(zip(predictions['top5_classes'][:5], predictions['top5_scores'][:5])):
                text = f"  {i+1}. Class {cls}: {score:.3f}"
                cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # For detection models
        elif 'boxes' in predictions:
            boxes = predictions.get('boxes', [])
            scores = predictions.get('scores', [])
            classes = predictions.get('classes', [])
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"Class {cls}: {score:.2f}"
                cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image
    
    def _draw_metrics(self, width: int, height: int, metrics: ComparisonMetrics) -> np.ndarray:
        """Draw metrics as text on black background"""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_offset = 50
        line_height = 30
        
        # Calculate overall quality (same as original code)
        avg_similarity = np.mean([metrics.cosine_similarity, metrics.pearson_correlation, metrics.ssim])
        
        if avg_similarity > 0.99:
            quality = "EXCELLENT"
        elif avg_similarity > 0.95:
            quality = "GOOD"
        elif avg_similarity > 0.90:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
        
        texts = [
            f"Cosine Sim: {metrics.cosine_similarity:.6f}",
            f"Pearson: {metrics.pearson_correlation:.6f}",
            f"MSE: {metrics.mse:.6f}",
            f"MAE: {metrics.mae:.6f}",
            f"Norm RMSE: {metrics.normalized_rmse:.6f}",
            f"SSIM: {metrics.ssim:.6f}",
            f"Rel Error: {metrics.relative_error:.2f}%",
            "",
            f"Assessment: {quality}",
        ]
        
        for i, text in enumerate(texts):
            if text == "":
                y_offset += line_height // 2
                continue
            
            # Highlight assessment
            color = (0, 255, 255) if i == len(texts) - 1 else (255, 255, 255)
            weight = 2 if i == len(texts) - 1 else 1
            
            cv2.putText(canvas, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, weight)
            y_offset += line_height
        
        return canvas
    
    def compare(
        self,
        output_path: Optional[str] = None,
        show_visualization: bool = False
    ) -> ComparisonMetrics:
        """
        Main comparison workflow
        
        Args:
            output_path: Path to save visualization
            show_visualization: Whether to display visualization
            
        Returns:
            ComparisonMetrics
        """
        print(f"\n{'='*60}")
        print(f"Running Model Comparison")
        print(f"{'='*60}\n")
        
        # Run ONNX inference
        print(f"\n{'='*60}")
        print(f"Running ONNX Runtime Inference")
        print(f"{'='*60}")
        onnx_preprocessed, onnx_raw, onnx_output = self.run_inference(SessionType.onnxruntime)
        
        # Run DXNN inference
        print(f"\n{'='*60}")
        print(f"Running DX Runtime (NPU) Inference")
        print(f"{'='*60}")
        dxnn_preprocessed, dxnn_raw, dxnn_output = self.run_inference(SessionType.dxruntime)
        
        # Calculate metrics
        metrics = self.calculate_metrics(onnx_raw, dxnn_raw)
        metrics.print()
        
        # Create visualization
        if output_path or show_visualization:
            vis = self.visualize_comparison(self.original_image, onnx_output, dxnn_output, metrics, output_path)
            
            if show_visualization:
                cv2.imshow("ONNX vs DXNN Comparison", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compare ONNX vs DXNN model outputs using dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare YOLOv5 models (uses first image from COCO dataset)
  python -m dx_modelzoo.tools.model_comparison \\
      --model yolov5 \\
      --onnx models/yolov5.onnx \\
      --dxnn models/yolov5.dxnn \\
      --data-dir /path/to/coco \\
      --output comparison.png

  # Compare with specific image index
  python -m dx_modelzoo.tools.model_comparison \\
      --model resnet50 \\
      --onnx models/resnet50.onnx \\
      --dxnn models/resnet50.dxnn \\
      --data-dir /path/to/imagenet \\
      --index 5 \\
      --show
        """
    )
    
    parser.add_argument('--model', required=True, help='Model name (e.g., YOLOV5N, ResNet50)')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model file')
    parser.add_argument('--dxnn', required=True, help='Path to DXNN model file')
    parser.add_argument('--data-dir', required=True, help='Dataset directory (required for evaluator)')
    parser.add_argument('--index', type=int, default=0, help='Image index from dataset (default: 0)')
    parser.add_argument('--output', '-o', help='Path to save comparison visualization')
    parser.add_argument('--show', action='store_true', help='Display visualization window')
    
    args = parser.parse_args()
    
    try:
        # Create comparator
        comparator = ModelComparator(
            model_name=args.model,
            onnx_path=args.onnx,
            dxnn_path=args.dxnn,
            data_dir=args.data_dir,
            image_index=args.index
        )
        
        # Run comparison
        metrics = comparator.compare(
            output_path=args.output,
            show_visualization=args.show
        )
        
        print("\n✓ Comparison completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
