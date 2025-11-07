#!/usr/bin/env python3
"""
FIFA 2026 Predictor - One-Click Run Script
==========================================

This script provides a single command to run the entire FIFA 2026 prediction pipeline:
1. Install dependencies
2. Initialize data collection
3. Launch the Streamlit dashboard

Usage:
    python run.py

Or make it executable and run:
    chmod +x run.py
    ./run.py
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description, cwd=None):
    """Run a shell command and handle errors."""
    logger.info(f"🔄 {description}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    """Main execution function."""
    logger.info("🚀 Starting FIFA 2026 Predictor Setup")
    logger.info("=" * 50)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        logger.error("❌ Please run this script from the project root directory (where pyproject.toml is located)")
        sys.exit(1)

    # Step 1: Install dependencies
    logger.info("\n📦 Step 1: Installing dependencies")
    if not run_command("uv sync", "Installing Python dependencies with uv"):
        logger.error("❌ Dependency installation failed. Please check your uv installation.")
        sys.exit(1)

    # Step 2: Initialize data collection
    logger.info("\n📊 Step 2: Initializing data collection")
    if not run_command("python initialize_data.py", "Running data initialization script"):
        logger.warning("⚠️ Data initialization had some issues, but continuing...")

    # Step 3: Launch Streamlit app
    logger.info("\n🌐 Step 3: Launching Streamlit dashboard")
    logger.info("📱 The app will be available at: http://localhost:5000")
    logger.info("🛑 Press Ctrl+C to stop the server")

    try:
        # Run streamlit in the foreground
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "5000",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
