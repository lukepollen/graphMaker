import subprocess
import sys

# Clone the repository
subprocess.run(["git", "clone", "https://github.com/neuml/txtai.git"], check=True)

# Navigate into the directory and install with extras
subprocess.run(["pip", "install", "./txtai[graph,autoawq]"], check=True)

# Verify the installation
try:
    import txtai
    print("txtai version:", txtai.__version__)
except ImportError:
    print("Failed to install txtai.")












