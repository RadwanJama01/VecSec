"""
Minimal Hugging Face Inference Test - One embedding
Just verify it works.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path (before any imports)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Suppress metrics initialization noise
os.environ.setdefault("METRICS_ENABLED", "false")

print("🔍 Testing Hugging Face Inference - One embedding\n")

try:
    from huggingface_hub import InferenceClient
    import numpy as np
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set")
        print("   Set it with: export HF_TOKEN=your_token")
        sys.exit(1)
    
    # Create client
    client = InferenceClient(
        provider="nebius",
        api_key=token
    )
    
    # ONE embedding call
    text = "Test embedding query"
    print(f"   Input: '{text}'")
    print(f"   Model: Qwen/Qwen3-Embedding-8B")
    
    result = client.feature_extraction(text, model="Qwen/Qwen3-Embedding-8B")
    
    # Extract embedding
    if isinstance(result, np.ndarray):
        if result.ndim == 2:
            embedding = result[0]
        else:
            embedding = result
    elif isinstance(result, list) and len(result) > 0:
        embedding = result[0] if isinstance(result[0], (list, np.ndarray)) else result
    else:
        embedding = result
    
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    print(f"\n✅ Hugging Face inference works!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Install with: pip install huggingface_hub numpy")
except Exception as e:
    error_msg = str(e)
    if "402" in error_msg or "Payment Required" in error_msg:
        print(f"⚠️  Payment Required (402): Monthly quota exceeded")
    else:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
