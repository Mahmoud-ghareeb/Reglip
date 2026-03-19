"""Example script demonstrating RegLIP usage."""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Import RegLIP components
from reglip import RegLIPModel, RegLIPConfig
from reglip.utils import load_from_transformers_siglip, setup_frozen_text_encoder


def example_basic_usage():
    """Basic example of creating and using RegLIP model."""
    print("=== Basic RegLIP Usage ===")
    
    # Create a RegLIP model with default configuration
    config = RegLIPConfig()
    model = RegLIPModel(config)
    
    print(f"Created RegLIP model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example inputs
    batch_size = 4
    text_length = 16
    image_size = 224
    
    # Dummy text inputs (in practice, use a tokenizer)
    input_ids = torch.randint(0, 1000, (batch_size, text_length))
    attention_mask = torch.ones(batch_size, text_length)
    
    # Dummy image inputs
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        return_loss=True,
    )
    
    print(f"Text embeddings shape: {outputs.text_embeds.shape}")
    print(f"Image embeddings shape: {outputs.image_embeds.shape}")
    print(f"Logits per text shape: {outputs.logits_per_text.shape}")
    print(f"Loss: {outputs.loss.item():.4f}")


def example_load_from_siglip():
    """Example of loading RegLIP from pre-trained SigLIP."""
    print("\n=== Loading from SigLIP ===")
    
    try:
        # Load a SigLIP model and convert to RegLIP
        model, config = load_from_transformers_siglip("google/siglip-base-patch16-224")
        
        print(f"Successfully loaded SigLIP model and converted to RegLIP")
        print(f"Text config: hidden_size={config.text_config.hidden_size}, layers={config.text_config.num_hidden_layers}")
        print(f"Vision config: hidden_size={config.vision_config.hidden_size}, layers={config.vision_config.num_hidden_layers}")
        
        # Set up frozen text encoder for similarity targets
        setup_frozen_text_encoder(model)
        
        return model, config
        
    except ImportError as e:
        print(f"Could not load from transformers: {e}")
        print("Install transformers and sentence-transformers to use this feature")
        return None, None


def example_regression_loss():
    """Demonstrate the regression-based contrastive loss."""
    print("\n=== Regression-based Loss ===")
    
    # Create model
    config = RegLIPConfig()
    model = RegLIPModel(config)
    
    # Example batch
    batch_size = 3
    
    # Create example similarity targets (continuous values between 0 and 1)
    similarity_targets = torch.tensor([
        [1.0, 0.7, 0.2],  # Text 0: high similarity to itself, medium to text 1, low to text 2
        [0.7, 1.0, 0.3],  # Text 1: medium to text 0, high to itself, low to text 2  
        [0.2, 0.3, 1.0],  # Text 2: low to others, high to itself
    ])
    
    # Example logits
    logits_per_text = torch.randn(batch_size, batch_size, requires_grad=True)
    
    # Compute regression loss
    loss = model.regression_contrastive_loss(logits_per_text, similarity_targets)
    
    print(f"Example similarity targets:\n{similarity_targets}")
    print(f"Predicted similarities (after sigmoid):\n{torch.sigmoid(logits_per_text).detach()}")
    print(f"Regression loss: {loss.item():.4f}")
    
    # Show that loss can be backpropagated
    loss.backward()
    print(f"Gradients computed successfully: {logits_per_text.grad is not None}")


def example_feature_extraction():
    """Example of feature extraction using RegLIP."""
    print("\n=== Feature Extraction Example ===")
    
    # Create model
    config = RegLIPConfig()
    model = RegLIPModel(config)
    
    # Create dummy inputs
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 16))
    attention_mask = torch.ones(batch_size, 16)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    
    # Extract features
    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_features = model.get_image_features(
            pixel_values=pixel_values
        )
    
    print(f"Text features shape: {text_features.shape}")
    print(f"Image features shape: {image_features.shape}")
    
    # Features are normalized and ready for similarity computation
    similarity = torch.matmul(text_features, image_features.t())
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity scores: {similarity}")


def example_embedding_pipeline():
    """Example using the embedding pipeline for text similarity."""
    print("\n=== Embedding Pipeline Example ===")
    
    try:
        from reglip.embedding_utils import QwenEmbeddingClient, batch_embedding_pipeline
        
        # Note: This requires a running Qwen embedding server
        print("Example of embedding pipeline usage:")
        print("1. Set up QwenEmbeddingClient with your API endpoint")
        print("2. Use batch_embedding_pipeline for text similarity")
        print("3. Get similarity predictions for text classification")
        
        sample_code = '''
# Initialize client
client = QwenEmbeddingClient(
    base_url="http://your-server:port",
    model="Qwen/Qwen3-Embedding-8B"
)

# Define texts and labels
texts = [
    "hello my name is mahmoud",
    "I love machine learning and AI",
    "The weather is beautiful today"
]

labels = ["greeting", "technology", "weather"]

# Run pipeline
text_embeddings, label_embeddings, predictions = batch_embedding_pipeline(
    texts=texts,
    labels=labels,
    client=client,
    batch_size=4,
    similarity_metric="dot_product",
    normalize=True,
    top_k=2
)
'''
        print(sample_code)
        
    except ImportError:
        print("Embedding utilities not available (requires additional dependencies)")


def example_model_loading_test():
    """Test RegLIP model loading and basic functionality."""
    print("\n=== Model Loading Test ===")
    
    try:
        # Test basic model creation
        config = RegLIPConfig()
        model = RegLIPModel(config)
        
        print(f"✅ Successfully created RegLIP model")
        print(f"   - Text config: hidden_size={config.text_config.hidden_size}")
        print(f"   - Vision config: hidden_size={config.vision_config.hidden_size}")
        
        # Test forward pass
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 16))
        attention_mask = torch.ones(batch_size, 16)
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=False
            )
            
        print(f"✅ Forward pass successful!")
        print(f"   - Output type: {type(outputs)}")
        print(f"   - Has text_embeds: {hasattr(outputs, 'text_embeds')}")
        print(f"   - Has image_embeds: {hasattr(outputs, 'image_embeds')}")
        print(f"   - Text embeds shape: {outputs.text_embeds.shape}")
        print(f"   - Image embeds shape: {outputs.image_embeds.shape}")
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")


def main():
    """Run all examples."""
    print("RegLIP Example Script")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    example_basic_usage()
    
    model, config = example_load_from_siglip()
    
    example_regression_loss()
    
    example_feature_extraction()
    
    example_embedding_pipeline()
    
    example_model_loading_test()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    
    if model is not None:
        print("\nTo use the loaded SigLIP model for training:")
        print("1. Set up your dataset with image-text pairs")
        print("2. Use the frozen text encoder to generate similarity targets")
        print("3. Train with the regression_contrastive_loss")
        print("4. Compare performance against the original binary loss")


if __name__ == "__main__":
    main() 