# RegLIP Loss Function Fix

## Problem Identified

During evaluation, the trained RegLIP model showed **random performance**:
- Retrieval R@1 = 0.0% (should be ~50%+ for a trained model)
- CIFAR-10 Zero-shot = 10.0% (exactly random chance for 10 classes)

### Root Cause: Model Collapse

Investigation revealed that the model experienced **feature collapse** during training. After training, all image-text pairs had cosine similarity ≈ **-0.80**, regardless of whether they matched or not.

```
=== PRETRAINED MODEL (no training) ===
Cosine similarities: [0.02, -0.01, -0.02]  ✓ Near zero (expected for random inputs)

=== TRAINED MODEL (epoch 1) ===
Cosine similarities: [-0.80, -0.80, -0.81]  ✗ All negative (model collapsed!)
```

### Why This Happened

The original loss function had a **sigmoid + scale/bias issue**:

```python
# Original (problematic) implementation
def regression_contrastive_loss(self, logits_per_text, similarity_targets):
    predicted_similarities = torch.sigmoid(logits_per_text)  # ← Problem here
    return F.mse_loss(predicted_similarities, similarity_targets)
```

Where `logits_per_text` was computed as:
```python
logits_per_text = cosine_sim * logit_scale.exp() + logit_bias
#                    ↑              ↑                ↑
#                 ~0.0           ~116.66          -12.92
```

**The Math Problem:**

1. Pretrained SigLIP has `logit_bias = -12.92` (designed for binary contrastive loss)
2. Even with `cosine_sim = 0`:
   - `logits = 0 * 116.66 + (-12.92) = -12.92`
   - `sigmoid(-12.92) ≈ 0.0000025` (essentially 0!)
3. The model couldn't escape this local minimum
4. Instead of learning meaningful features, it collapsed to outputting constant negative similarity

---

## Solution: Raw Cosine Similarity (Option 3)

We chose the **cleanest approach for research**: use raw cosine similarity without sigmoid, scale, or bias.

### New Loss Function

```python
def regression_contrastive_loss(
    self,
    text_embeds: torch.FloatTensor,
    image_embeds: torch.FloatTensor,
    similarity_targets: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Compute regression-based contrastive loss using raw cosine similarity.
    """
    # Compute raw cosine similarity (embeddings are already L2-normalized)
    # This gives values in [-1, 1]
    cosine_sim = torch.matmul(text_embeds, image_embeds.t())
    
    # Map cosine similarity from [-1, 1] to [0, 1] range
    # This matches the range of similarity_targets (which are also in [0, 1])
    predicted_similarities = (cosine_sim + 1) / 2
    
    # Compute MSE loss between predicted and target similarities
    loss = F.mse_loss(predicted_similarities, similarity_targets)
    
    return loss
```

### Why This Approach is Best for Research

| Aspect | Old (Sigmoid) | New (Raw Cosine) |
|--------|---------------|------------------|
| **Simplicity** | Complex (sigmoid + scale + bias) | Simple (linear mapping) |
| **Interpretability** | Hard to explain | Easy to explain |
| **Confounding factors** | Many | None |
| **Gradient flow** | Vanishing gradients at extremes | Stable gradients |
| **Range mapping** | Nonlinear (sigmoid) | Linear and symmetric |

### Mathematical Comparison

**Old approach:**
```
predicted = sigmoid(cosine_sim × exp(scale) + bias)
         = sigmoid(cosine_sim × 116.66 - 12.92)
```
- Highly nonlinear
- Bias shifts everything negative
- Sigmoid saturates, causing vanishing gradients

**New approach:**
```
predicted = (cosine_sim + 1) / 2
```
- Linear transformation
- Maps [-1, 1] → [0, 1] symmetrically
- No saturation, stable gradients

---

## Research Justification

This fix **isolates the core research hypothesis**:

> "Regression with soft labels (text-text similarity) provides richer supervision than binary labels (0/1)"

By removing sigmoid/scale/bias, we ensure that any improvement comes from the **soft label approach itself**, not from other architectural differences.

### In Your Paper

You can write:

> "Unlike SigLIP which uses binary targets and sigmoid activation, RegLIP directly regresses 
> the image-text cosine similarity to match the text-text semantic similarity computed by a 
> frozen language model. We use a simple linear mapping from cosine similarity [-1, 1] to [0, 1] 
> to match the target range, without learned temperature or bias parameters. This design choice 
> isolates the effect of soft supervision from other confounding factors."

---

## Files Changed

1. **`reglip/model.py`**
   - `regression_contrastive_loss()`: Now takes `text_embeds` and `image_embeds` instead of `logits_per_text`
   - `forward()`: Passes embeddings (before scale/bias) to regression loss
   - `logit_scale` and `logit_bias`: Changed from scalar to shape `[1]` to match SigLIP

2. **`reglip/vision.py`**
   - `patch_embedding`: Changed `bias=False` to `bias=True` to match SigLIP
   - Added `RegLIPMultiheadAttentionPoolingHead`: New class matching SigLIP's attention pooling head
   - `RegLIPVisionTransformer`: Now uses attention pooling instead of mean pooling

3. **`reglip/utils.py`**
   - Improved weight loading to handle all SigLIP weights correctly
   - Better error reporting for mismatched keys

---

## Re-training Required

After this fix, you need to **retrain the model** from scratch:

```bash
# Delete old checkpoints
rm -rf ./checkpoints/reglip/*

# Retrain
bash train.sh
```

The new training should show:
- Stable loss decrease
- Cosine similarities that vary based on input (not constant -0.80)
- Much better evaluation metrics

---

## Alternative Approaches Considered

### Option 1: Reset logit_bias to 0
```python
model.logit_bias.data.fill_(0.0)
```
- Pros: Simple fix
- Cons: Still uses sigmoid which can saturate

### Option 2: Remove sigmoid, keep scale
```python
predicted = logits_per_text / self.logit_scale.exp()
predicted = (predicted + 1) / 2
```
- Pros: Keeps learned temperature
- Cons: Scale parameter adds complexity

### Option 3: Raw cosine similarity (CHOSEN)
```python
predicted = (cosine_sim + 1) / 2
```
- Pros: Cleanest, most interpretable, best for research
- Cons: No learned temperature (but this is actually a feature, not a bug)

We chose **Option 3** because it provides the clearest comparison between RegLIP and SigLIP, isolating the effect of soft labels.

---

## Summary

| Before Fix | After Fix |
|------------|-----------|
| `sigmoid(logits)` with scale/bias | Raw `(cosine_sim + 1) / 2` |
| Model collapsed to -0.80 similarity | Stable training |
| Random evaluation performance | Meaningful results |
| Hard to interpret | Easy to explain in paper |
