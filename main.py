import argparse
import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(command: list, step_name: str):
    logger.info(f"STARTING: {step_name}")
    try:
        # Run module as script
        result = subprocess.run(command, check=True, capture_output=False)
        logger.info(f"COMPLETED: {step_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {step_name} with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Real Madrid Pressing Analysis E2E Pipeline")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-models", action="store_true", help="Skip modeling")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()
    
    python_exe = sys.executable
    
    # 1. Feature Extraction
    if not args.skip_features:
        run_step([python_exe, "-m", "src.features.feature_engineering", "--verbose"], "Feature Engineering")
        
    # 2. Modeling
    if not args.skip_models:
        # GMM
        run_step([python_exe, "-m", "src.models.gmm_zones"], "GMM Zone Fitting")
        
        # Tokenization
        run_step([python_exe, "-m", "src.models.tokenization"], "Tokenization")
        
        # NMF
        run_step([python_exe, "-m", "src.models.nmf_topics", "--n-topics", "15"], "NMF Topic Modeling")
        
        # Clustering
        run_step([python_exe, "-m", "src.models.clustering", "--n-clusters", "5"], "Clustering")
        
    # 3. Visualization
    if not args.skip_viz:
        run_step([python_exe, "-m", "src.viz.generate_all", "--out-dir", "visualizations"], "Visualization Generation")

    logger.info("PIPELINE COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
