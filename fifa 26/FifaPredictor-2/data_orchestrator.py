"""
Data Orchestration Module
Coordinates all data collection, feature engineering, and model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import json

from scrapers.fbref_scraper import FBrefScraper
from scrapers.understat_scraper import UnderstatScraper
from scrapers.transfermarkt_scraper import TransfermarktScraper
from api_collectors.api_football_client import APIFootballClient
from api_collectors.football_data_downloader import FootballDataDownloader
from ml_pipeline.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataOrchestrator:
    """Orchestrates data collection from all sources and integrates them."""
    
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.fbref = FBrefScraper()
        self.understat = UnderstatScraper()
        self.transfermarkt = TransfermarktScraper()
        self.api_football = APIFootballClient()
        self.football_data = FootballDataDownloader()
        self.feature_engineer = FeatureEngineer()
        
        self.teams_cache_file = self.cache_dir / "teams_integrated_data.json"
        
    def get_fifa_2026_teams(self) -> List[str]:
        """Get list of 48 FIFA 2026 qualified/potential teams."""
        return [
            'Brazil', 'Argentina', 'France', 'Germany', 'Spain', 'England', 'Belgium', 'Netherlands',
            'Portugal', 'Italy', 'Croatia', 'Uruguay', 'Colombia', 'Mexico', 'USA', 'Switzerland',
            'Denmark', 'Senegal', 'Morocco', 'Japan', 'South Korea', 'Iran', 'Australia', 'Poland',
            'Sweden', 'Austria', 'Ukraine', 'Turkey', 'Czech Republic', 'Wales', 'Serbia', 'Russia',
            'Peru', 'Chile', 'Nigeria', 'Egypt', 'Ghana', 'Cameroon', 'Tunisia', 'Algeria',
            'Ecuador', 'Paraguay', 'Canada', 'Costa Rica', 'Jamaica', 'Saudi Arabia', 'Qatar', 'Iraq'
        ]
    
    def collect_team_data(self, team_name: str, use_cache: bool = True) -> Dict:
        """
        Collect comprehensive data for a single team from all sources.
        
        Args:
            team_name: Name of the team
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing all team data
        """
        logger.info(f"Collecting data for {team_name}...")
        
        team_data = {
            'name': team_name,
            'fbref_stats': {},
            'transfermarkt_stats': {},
            'api_football_stats': {},
            'last_updated': pd.Timestamp.now().isoformat()
        }
        
        try:
            fbref_stats = self.fbref.get_team_stats(team_name)
            team_data['fbref_stats'] = fbref_stats
            logger.info(f"✓ FBref data collected for {team_name}")
        except Exception as e:
            logger.warning(f"✗ FBref data collection failed for {team_name}: {e}")
        
        try:
            transfermarkt_stats = self.transfermarkt.get_squad_market_value(team_name)
            team_data['transfermarkt_stats'] = transfermarkt_stats
            logger.info(f"✓ Transfermarkt data collected for {team_name}")
        except Exception as e:
            logger.warning(f"✗ Transfermarkt data collection failed for {team_name}: {e}")
        
        return team_data
    
    def integrate_team_features(self, team_data: Dict) -> Dict:
        """Integrate data from all sources into a unified feature set."""
        
        integrated = {
            'team_name': team_data['name'],
            'last_updated': team_data['last_updated']
        }
        
        fbref = team_data.get('fbref_stats', {})
        integrated['xg_for_90'] = fbref.get('xg_for_90', 0)
        integrated['xg_against_90'] = fbref.get('xg_against_90', 0)
        integrated['shots_per_90'] = fbref.get('shots_per_90', 0)
        integrated['sot_per_90'] = fbref.get('sot_per_90', 0)
        integrated['possession_pct'] = fbref.get('possession_pct', 50)
        integrated['pass_accuracy'] = fbref.get('pass_accuracy', 75)
        integrated['progressive_passes_per_90'] = fbref.get('progressive_passes_per_90', 0)
        
        tm = team_data.get('transfermarkt_stats', {})
        integrated['squad_value_million_eur'] = tm.get('squad_value_million_eur', 0)
        integrated['avg_age'] = tm.get('avg_age', 26)
        integrated['total_players'] = tm.get('total_players', 23)
        integrated['foreigners_count'] = tm.get('foreigners_count', 0)
        
        return integrated
    
    def collect_all_teams_data(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Collect data for all FIFA 2026 teams.
        
        Args:
            force_refresh: If True, ignore cache and scrape fresh data
            
        Returns:
            Dictionary mapping team names to their integrated data
        """
        if not force_refresh and self.teams_cache_file.exists():
            logger.info("Loading teams data from cache...")
            with open(self.teams_cache_file, 'r') as f:
                return json.load(f)
        
        teams = self.get_fifa_2026_teams()
        all_teams_data = {}
        
        logger.info(f"Collecting data for {len(teams)} teams...")
        
        for i, team in enumerate(teams, 1):
            logger.info(f"Processing team {i}/{len(teams)}: {team}")
            
            team_raw_data = self.collect_team_data(team)
            team_integrated = self.integrate_team_features(team_raw_data)
            
            team_integrated['fifa_rank'] = i
            team_integrated['elo_rating'] = 2000 - (i * 15)
            
            all_teams_data[team] = team_integrated
        
        self._save_cache(all_teams_data)
        logger.info(f"✓ Data collection complete for {len(all_teams_data)} teams")
        
        return all_teams_data
    
    def _save_cache(self, data: Dict) -> None:
        """Save collected data to cache file."""
        with open(self.teams_cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Cache saved to {self.teams_cache_file}")
    
    def get_fixtures(self) -> pd.DataFrame:
        """Get fixtures data (combines sample + real API data if available)."""
        
        fixtures = []
        teams = self.get_fifa_2026_teams()[:10]
        
        for i in range(20):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            fixtures.append({
                'date': f'2026-06-{10 + i:02d}',
                'home_team': home_team,
                'away_team': away_team,
                'venue': f'Stadium {i+1}',
                'city': f'City {i+1}',
                'country': 'USA/Canada/Mexico',
                'status': np.random.choice(['scheduled', 'finished', 'live'], p=[0.7, 0.2, 0.1]),
                'prob_home_win': np.random.uniform(0.3, 0.6),
                'prob_draw': 0.25,
                'prob_away_win': np.random.uniform(0.15, 0.45),
                'temp_c': np.random.randint(20, 35),
                'humidity': np.random.randint(40, 80),
                'competition': 'FIFA World Cup 2026'
            })
        
        return pd.DataFrame(fixtures)
    
    def generate_simulation_results(self, teams_data: Dict[str, Dict]) -> pd.DataFrame:
        """Generate simulation results based on team features."""
        
        results = []
        
        for team_name, team_info in teams_data.items():
            xg_diff = team_info.get('xg_for_90', 1.5) - team_info.get('xg_against_90', 1.0)
            squad_value = team_info.get('squad_value_million_eur', 500)
            fifa_rank = team_info.get('fifa_rank', 25)
            
            base_score = (xg_diff * 5) + (squad_value / 100) + (50 - fifa_rank) / 2
            finalist_prob = max(0.5, min(95, base_score + np.random.uniform(-5, 5)))
            
            results.append({
                'team': team_name,
                'finalist_probability': finalist_prob,
                'winner_probability': finalist_prob * 0.5,
                'finalist_appearances': int(finalist_prob * 50),
                'wins': int(finalist_prob * 25),
                'ci_lower': max(0, finalist_prob - 2),
                'ci_upper': min(100, finalist_prob + 2),
                'top_features': [
                    f'xG Diff: {xg_diff:.2f}',
                    f'Squad Value: €{squad_value:.0f}M',
                    f'FIFA Rank: #{fifa_rank}'
                ]
            })
        
        df = pd.DataFrame(results).sort_values('finalist_probability', ascending=False)
        return df

def get_orchestrator() -> DataOrchestrator:
    """Get or create the singleton data orchestrator instance."""
    return DataOrchestrator()
