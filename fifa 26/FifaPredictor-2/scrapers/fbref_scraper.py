import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import Dict, List, Optional
import logging
from requests_cache import CachedSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FBrefScraper:
    """Scrapes team statistics from FBref including xG, shots, possession, and passing data."""
    
    def __init__(self, cache_expire_after=3600):
        self.base_url = "https://fbref.com"
        self.session = CachedSession('fbref_cache', expire_after=cache_expire_after)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_team_stats(self, team_name: str, competition: str = "World-Cup") -> Dict:
        """
        Scrape team statistics from FBref.
        
        Args:
            team_name: Name of the team
            competition: Competition name (default: World-Cup)
            
        Returns:
            Dictionary containing team statistics
        """
        try:
            time.sleep(3)
            
            search_url = f"{self.base_url}/en/search/search.fcgi?search={team_name.replace(' ', '+')}"
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            stats = {
                'team_name': team_name,
                'xg_for_90': 0.0,
                'xg_against_90': 0.0,
                'shots_per_90': 0.0,
                'sot_per_90': 0.0,
                'possession_pct': 0.0,
                'pass_accuracy': 0.0,
                'progressive_passes_per_90': 0.0,
                'data_source': 'fbref',
                'last_updated': pd.Timestamp.now()
            }
            
            stats_table = soup.find('table', {'id': 'stats_standard'})
            if stats_table:
                rows = stats_table.find('tbody').find_all('tr')
                if rows:
                    cells = rows[0].find_all('td')
                    if len(cells) >= 15:
                        try:
                            stats['xg_for_90'] = float(cells[7].text.strip() or 0) if len(cells) > 7 else 0.0
                            stats['shots_per_90'] = float(cells[10].text.strip() or 0) if len(cells) > 10 else 0.0
                            stats['sot_per_90'] = float(cells[11].text.strip() or 0) if len(cells) > 11 else 0.0
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing stats for {team_name}: {e}")
            
            possession_table = soup.find('table', {'id': 'stats_possession'})
            if possession_table:
                rows = possession_table.find('tbody').find_all('tr')
                if rows:
                    cells = rows[0].find_all('td')
                    try:
                        stats['possession_pct'] = float(cells[2].text.strip().replace('%', '') or 0) if len(cells) > 2 else 0.0
                    except (ValueError, IndexError):
                        pass
            
            passing_table = soup.find('table', {'id': 'stats_passing'})
            if passing_table:
                rows = passing_table.find('tbody').find_all('tr')
                if rows:
                    cells = rows[0].find_all('td')
                    try:
                        stats['pass_accuracy'] = float(cells[5].text.strip().replace('%', '') or 0) if len(cells) > 5 else 0.0
                        stats['progressive_passes_per_90'] = float(cells[15].text.strip() or 0) if len(cells) > 15 else 0.0
                    except (ValueError, IndexError):
                        pass
            
            logger.info(f"Successfully scraped FBref stats for {team_name}")
            return stats
            
        except Exception as e:
            logger.error(f"Error scraping FBref for {team_name}: {e}")
            return {
                'team_name': team_name,
                'xg_for_90': None,
                'xg_against_90': None,
                'shots_per_90': None,
                'sot_per_90': None,
                'possession_pct': None,
                'pass_accuracy': None,
                'progressive_passes_per_90': None,
                'data_source': 'fbref',
                'last_updated': pd.Timestamp.now(),
                'error': str(e)
            }
    
    def get_league_stats(self, league_url: str) -> pd.DataFrame:
        """
        Scrape all team statistics from a league page.
        
        Args:
            league_url: Full URL to the league stats page
            
        Returns:
            DataFrame containing stats for all teams
        """
        try:
            time.sleep(3)
            response = self.session.get(league_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            tables = pd.read_html(response.content)
            
            if tables:
                df = tables[0]
                logger.info(f"Successfully scraped league stats: {len(df)} teams")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error scraping league stats: {e}")
            return pd.DataFrame()
