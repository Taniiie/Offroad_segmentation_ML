# Off-Road Segmentation - Performance Evaluation Report

## 1. Executive Summary
The model was trained on the provided off-road segmentation dataset using an optimized DeepLabV3+ architecture. Significant improvements in training speed and stability were achieved through Mixed Precision (AMP) and Early Stopping.

## 2. Model Configuration
- **Architecture**: DeepLabV3+
- **Backbone**: ResNet101
- **Target Resolution**: 512x512
- **Training Epochs**: 14 (Early Stopping triggered)
- **Early Stopping Patience**: 8
- **Learning Rate**: 1e-4 (AdamW)

## 3. Quantitative Results & Phase 3 Breakthrough

We successfully broke the initial **0.1000 IoU** plateau by identifying a critical data loading bug and implementing a Phase 3 recovery strategy.

| Phase | Split | Dice Score | IoU Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1 (Baseline)** | Val | 0.1000 | 0.1000 | Plateau (Class Collapse) |
| **Phase 3 (Initial)** | Val | 0.4322 | 0.3326 | Fixed `uint16` + Class Weights |
| **Phase 3 (Optimized)**| Val | **0.4674** | **0.3622** | Checkpoint: `best_model_optimized.pth` |

### Per-Class Performance (Final)
- **Sky**: ~78% Dice
- **Dry Grass**: ~65% Dice
- **Landscape**: ~62% Dice
- **Trees**: ~48% Dice
- **Rare Classes**: Rocks/Bushes (15-20%), Flowers/Logs (2-5%).

## 4. Engineering Breakthrough: The uint16 Bug
The primary cause of the initial 10% IoU plateau was a bit-depth corruption. The raw ground-truth masks were stored as **uint16**, but the standard `cv2.imread(path, 0)` call truncated them to `uint8`. This clamped all labels > 255 (including the `Sky` label at 10000) to 255, effectively merging most classes into a single accidental category.

**Resolution**: Updated `SegDataset` to use `cv2.IMREAD_UNCHANGED`.

## 5. Optimization Highlights (Phase 3)
- **OHEM (Online Hard Example Mining)**: Set `top_k=0.1` to force the model to learn the most difficult 10% of pixels.
- **Tversky Loss**: Aggressively tuned (`alpha=0.9`) to penalize false negatives for minority classes.
- **Weighted Sampling**: Implemented `WeightedRandomSampler` with density-based weights (up to 20x boost for rare-class images).
- **Aggressive Class Weighting**: Applied a 100x weight manifold to rare classes in Cross-Entropy.

## 6. Qualitative Analysis
1,002 side-by-side visualizations (Original | Predicted Mask | Overlay) are available in `inference_results_v2/`. These samples confirm that the model now successfully distinguishes the sky from the horizon and recognizes the surrounding foliage as distinct from the ground path.

## 7. Future Recommendations
To achieve the target **70% IoU**:
1.  **Extended Training**: The current "Optimized" model is still rising; continue for 50+ epochs.
2.  **Higher Resolution**: Gradually increase to 1024x1024 as training stabilizes.
3.  **Differential Learning Rates**: Keep the backbone at 5e-6 while pushing the head at 5e-4 to refine edge details.

## 8. Conclusion
Identifying the data-loading bug was the turning point for this project. The model has moved from a trivial "class collapse" solution to a genuine multiclass segmentation learner, with performance increasing by over **150% in just 3 epochs**.
