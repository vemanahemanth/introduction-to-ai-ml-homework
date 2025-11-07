"""
Online Data Loader - Loads precomputed data for Streamlit app
"""

import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Loads precomputed data from ETL pipeline outputs."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.teams_file = self.data_dir / "teams_features.parquet"
        self.simulation_file = self.data_dir / "simulation_results.parquet"
        self.fixtures_file = self.data_dir / "fixtures.parquet"
        self.metadata_file = self.data_dir / "metadata.json"
    
    def load_metadata(self) -> Dict:
        """Load ETL metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'last_updated': 'Never',
            'num_teams': 0,
            'data_sources': [],
            'status': 'No data available'
        }
    
    def load_teams_data(self) -> Dict[str, Dict]:
        """Load team features from parquet file."""
        if self.teams_file.exists():
            df = pd.read_parquet(self.teams_file)
            teams_dict = df.to_dict('index')
            logger.info(f"✅ Loaded {len(teams_dict)} teams from {self.teams_file}")
            return teams_dict
        else:
            logger.warning(f"⚠️ Teams data file not found: {self.teams_file}")
            return {}
    
    def load_simulation_results(self) -> pd.DataFrame:
        """Load precomputed simulation results."""
        if self.simulation_file.exists():
            df = pd.read_parquet(self.simulation_file)
            logger.info(f"✅ Loaded simulation results: {len(df)} teams")
            return df
        else:
            logger.warning(f"⚠️ Simulation file not found: {self.simulation_file}")
            return pd.DataFrame()
    
    def load_fixtures(self) -> pd.DataFrame:
        """Load fixtures data."""
        if self.fixtures_file.exists():
            df = pd.read_parquet(self.fixtures_file)
            logger.info(f"✅ Loaded {len(df)} fixtures")
            return df
        else:
            logger.warning(f"⚠️ Fixtures file not found: {self.fixtures_file}")
            return pd.DataFrame()
    
    def is_data_available(self) -> bool:
        """Check if precomputed data is available."""
        return self.teams_file.exists() and self.simulation_file.exists()
    
    def get_data_freshness(self) -> str:
        """Get human-readable data freshness."""
        metadata = self.load_metadata()
        last_updated = metadata.get('last_updated', 'Never')
        
        if last_updated == 'Never':
            return "No data collected yet"
        
        try:
            updated_dt = datetime.fromisoformat(last_updated)
            now = datetime.now()
            delta = now - updated_dt
            
            if delta.days == 0:
                hours = delta.seconds // 3600
                if hours == 0:
                    minutes = delta.seconds // 60
                    return f"{minutes} minutes ago"
                return f"{hours} hours ago"
            elif delta.days == 1:
                return "1 day ago"
            else:
                return f"{delta.days} days ago"
        except:
            return last_updated
    
    def trigger_refresh(self) -> str:
        """Trigger data refresh by running ETL script."""
        import subprocess
        
        try:
            result = subprocess.run(
                ['python', 'etl_runner.py', '--force-refresh'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                return "✅ Data refresh completed successfully!"
            else:
                return f"⚠️ Refresh failed: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "⏱️ Refresh timeout - data collection takes longer than expected"
        except Exception as e:
            return f"❌ Error triggering refresh: {e}"
