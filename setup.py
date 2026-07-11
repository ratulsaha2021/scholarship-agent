#!/usr/bin/env python3
"""Setup script for Scholarship Agent."""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])

def create_directories():
    """Create necessary directories."""
    dirs = [
        "resources/templates",
        "resources/saved_responses",
        "config"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")

def main():
    print("=== Scholarship Agent Setup ===\n")
    
    create_directories()
    print()
    
    try:
        install_dependencies()
        print("\n✓ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        print("Try running manually: pip install -r requirements.txt")
        return
    
    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("1. Edit resources/user_data.json with your information")
    print("2. Add your CV (cv.pdf or cv.docx) to resources/")
    print("3. Add research_interests.txt and skills.txt to resources/")
    print("4. Add targets to resources/targets.json")
    print("5. Run: python -m src.agent --setup")

if __name__ == "__main__":
    main()
