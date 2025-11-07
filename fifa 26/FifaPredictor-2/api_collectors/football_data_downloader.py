import requests
import pandas as pd
import logging
from typing import List, Optional
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballDataDownloader:
    """Downloads historical match data and betting odds from football-data.co.uk."""
    
    def __init__(self, data_dir: str = "data/raw/football_data"):
        self.base_url = "https://www.football-data.co.uk"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_season_data(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """
        Download season data for a specific league.
        
        Args:
            league: League code (E0=Premier League, SP1=La Liga, etc.)
            season: Season in format YYZZ (e.g., '2324' for 2023-24)
            
        Returns:
            DataFrame containing match results and odds
        """
        try:
            year_folder = f"20{season[:2]}{season[2:]}"
            url = f"{self.base_url}/mmz4281/{year_folder}/{league}.csv"
            
            logger.info(f"Downloading {league} {season} from {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            csv_path = self.data_dir / f"{league}_{season}.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Successfully downloaded {len(df)} matches for {league} {season}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {league} {season}: {e}")
            return None
    
    def download_multiple_seasons(self, leagues: List[str], seasons: List[str]) -> pd.DataFrame:
        """
        Download data for multiple leagues and seasons.
        
        Args:
            leagues: List of league codes
            seasons: List of season codes
            
        Returns:
            Combined DataFrame
        """
        all_data = []
        
        for league in leagues:
            for season in seasons:
                df = self.download_season_data(league, season)
                if df is not None:
                    df['league'] = league
                    df['season'] = season
                    all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined {len(combined_df)} total matches")
            return combined_df
        
        return pd.DataFrame()
    
    def download_world_cup_data(self) -> pd.DataFrame:
        """Download World Cup historical data."""
        try:
            url = f"{self.base_url}/new/WC.csv"
            
            logger.info(f"Downloading World Cup data from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            csv_path = self.data_dir / "world_cup_historical.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Successfully downloaded {len(df)} World Cup matches")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading World Cup data: {e}")
            return pd.DataFrame()
    
    def get_betting_odds_columns(self) -> List[str]:
        """Return list of betting odds columns typically available."""
        return [
            'B365H', 'B365D', 'B365A',
            'BWH', 'BWD', 'BWA',
            'IWH', 'IWD', 'IWA',
            'PSH', 'PSD', 'PSA',
            'WHH', 'WHD', 'WHA',
            'VCH', 'VCD', 'VCA'
        ]
    
    def extract_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and clean relevant features from raw football-data.co.uk CSV.
        
        Args:
            df: Raw DataFrame from football-data.co.uk
            
        Returns:
            Cleaned DataFrame with standardized columns
        """
        try:
            required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            
            if not all(col in df.columns for col in required_cols):
                logger.warning("Missing required columns in DataFrame")
                return pd.DataFrame()
            
            df_clean = df[required_cols].copy()
            
            odds_cols = self.get_betting_odds_columns()
            for col in odds_cols:
                if col in df.columns:
                    df_clean[col] = df[col]
            
            if 'HS' in df.columns:
                df_clean['HomeShots'] = df['HS']
            if 'AS' in df.columns:
                df_clean['AwayShots'] = df['AS']
            if 'HST' in df.columns:
                df_clean['HomeShotsOnTarget'] = df['HST']
            if 'AST' in df.columns:
                df_clean['AwayShotsOnTarget'] = df['AST']
            
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            
            logger.info(f"Extracted features for {len(df_clean)} matches")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()
