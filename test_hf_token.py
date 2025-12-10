#!/usr/bin/env python3
"""
Test script to verify HuggingFace token and pyannote access.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 60)
print("HuggingFace Token & Pyannote Access Diagnostic")
print("=" * 60)

# Check if token is loaded
hf_token = os.environ.get('HF_TOKEN')
print(f"\n1. HF_TOKEN from environment: {'✓ Found' if hf_token else '✗ Not found'}")
if hf_token:
    print(f"   Token starts with: {hf_token[:10]}...")
    print(f"   Token length: {len(hf_token)}")

# Try to verify token with HuggingFace Hub
print("\n2. Verifying token with HuggingFace Hub...")
try:
    from huggingface_hub import HfApi, login
    
    login(token=hf_token, add_to_git_credential=False)
    
    api = HfApi()
    user_info = api.whoami()
    print(f"   ✓ Token is valid!")
    print(f"   Logged in as: {user_info.get('name', 'Unknown')}")
    
except Exception as e:
    print(f"   ✗ Token verification failed: {e}")

# Check model access by trying to download config
print("\n3. Checking DOWNLOAD access to pyannote models...")
print("   (This checks if you've accepted the license terms)")

from huggingface_hub import hf_hub_download

models_to_check = [
    ("pyannote/speaker-diarization-3.1", "config.yaml"),
    ("pyannote/segmentation-3.0", "config.yaml"),
]

all_accessible = True
for model_id, filename in models_to_check:
    try:
        path = hf_hub_download(repo_id=model_id, filename=filename, token=hf_token)
        print(f"   ✓ {model_id} - Access granted")
    except Exception as e:
        all_accessible = False
        error_msg = str(e)
        if "gated" in error_msg.lower() or "403" in error_msg:
            print(f"   ✗ {model_id}")
            print(f"     → You need to accept terms at: https://huggingface.co/{model_id}")
        else:
            print(f"   ✗ {model_id} - Error: {e}")

if not all_accessible:
    print("\n" + "!" * 60)
    print("ACTION REQUIRED:")
    print("  1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("  2. Click 'Agree and access repository' (must be logged in)")
    print("  3. Go to https://huggingface.co/pyannote/segmentation-3.0")
    print("  4. Click 'Agree and access repository'")
    print("  5. Re-run this test script")
    print("!" * 60)
else:
    # Try loading the pipeline
    print("\n4. Attempting to load pyannote pipeline...")
    try:
        from pyannote.audio import Pipeline
        
        print("   Loading pyannote/speaker-diarization-3.1...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        if pipeline is not None:
            print("   ✓ Pipeline loaded successfully!")
            print(f"   Pipeline type: {type(pipeline).__name__}")
        else:
            print("   ✗ Pipeline is None (unexpected)")
        
    except Exception as e:
        print(f"   ✗ Pipeline loading failed: {e}")

print("\n" + "=" * 60)

