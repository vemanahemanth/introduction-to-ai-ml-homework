#!/usr/bin/env python3
"""
Offline ETL Runner - Collects real data from all sources and caches for app consumption
Run this script independently to refresh data: python etl_runner.py
"""

import pandas as pd
import logging
from pathlib import Path
import argparse
from datetime import datetime

from data_orchestrator import DataOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='FIFA 2026 Data ETL Runner')
    parser.add_argument('--force-refresh', action='store_true', help='Force data refresh')
    parser.add_argument('--teams-only', action='store_true', help='Only collect team data')
    parser.add_argument('--sample-size', type=int, default=None, help='Limit to N teams for testing')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FIFA 2026 DATA ETL PIPELINE")
    logger.info("="*80)
    
    orchestrator = DataOrchestrator()
    
    logger.info("\n📊 Phase 1: Team Data Collection")
    logger.info("-" * 80)
    
    teams_data = orchestrator.collect_all_teams_data(force_refresh=args.force_refresh)
    logger.info(f"✅ Collected data for {len(teams_data)} teams")
    
    df_teams = pd.DataFrame.from_dict(teams_data, orient='index')
    teams_file = Path('data/processed/teams_features.parquet')
    teams_file.parent.mkdir(parents=True, exist_ok=True)
    df_teams.to_parquet(teams_file)
    logger.info(f"✅ Saved team features to {teams_file}")
    
    logger.info("\n🎲 Phase 2: Simulation Results Generation")
    logger.info("-" * 80)
    
    simulation_results = orchestrator.generate_simulation_results(teams_data)
    sim_file = Path('data/processed/simulation_results.parquet')
    simulation_results.to_parquet(sim_file)
    logger.info(f"✅ Saved simulation results to {sim_file}")
    
    if not args.teams_only:
        logger.info("\n📅 Phase 3: Fixtures Data")
        logger.info("-" * 80)
        
        fixtures_df = orchestrator.get_fixtures()
        fixtures_file = Path('data/processed/fixtures.parquet')
        fixtures_df.to_parquet(fixtures_file)
        logger.info(f"✅ Saved fixtures to {fixtures_file}")
    
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'num_teams': len(teams_data),
        'data_sources': ['FBref', 'Transfermarkt', 'API-Football', 'football-data.co.uk'],
        'etl_version': '1.0.0'
    }
    
    import json
    metadata_file = Path('data/processed/metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata to {metadata_file}")
    
    logger.info("\n" + "="*80)
    logger.info("ETL PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - {teams_file}")
    logger.info(f"  - {sim_file}")
    if not args.teams_only:
        logger.info(f"  - {fixtures_file}")
    logger.info(f"  - {metadata_file}")
    logger.info("\n✨ Data is ready for Streamlit app consumption\n")

if __name__ == "__main__":
    main()
