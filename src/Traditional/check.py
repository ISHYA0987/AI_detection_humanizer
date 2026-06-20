import importlib
import sys

# List of required libraries
libraries = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "datasets": "Datasets",
    "accelerate": "Accelerate",
    "evaluate": "Evaluate",
    "sklearn": "Scikit-learn",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "tqdm": "tqdm",
}

print("=" * 50)
print("Checking Required Libraries")
print("=" * 50)

missing = []

for module, name in libraries.items():
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "Unknown")
        print(f"✅ {name:<15} Installed (Version: {version})")
    except ImportError:
        print(f"❌ {name:<15} NOT Installed")
        missing.append(module)

print("=" * 50)

# CUDA Check
try:
    import torch

    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version    : {torch.version.cuda}")
    else:
        print("GPU             : CPU Only")

except ImportError:
    print("PyTorch not installed, skipping CUDA check.")

print("=" * 50)

if missing:
    print("Missing Libraries:")
    for lib in missing:
        print(f"  - {lib}")

    print("\nInstall them using:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("🎉 All required libraries are installed!")