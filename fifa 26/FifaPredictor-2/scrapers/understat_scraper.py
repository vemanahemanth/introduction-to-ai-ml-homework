import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import logging
from typing import Dict, List, Optional
from requests_cache import CachedSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnderstatScraper:
    """Scrapes xG and shot data from Understat."""
    
    def __init__(self, cache_expire_after=3600):
        self.base_url = "https://understat.com"
        self.session = CachedSession('understat_cache', expire_after=cache_expire_after)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_team_xg_data(self, team_name: str, league: str = "EPL", season: str = "2024") -> Dict:
        """
        Scrape xG data for a specific team.
        
        Args:
            team_name: Name of the team
            league: League code (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)
            season: Season year
            
        Returns:
            Dictionary containing xG statistics
        """
        try:
            time.sleep(2)
            
            team_slug = team_name.lower().replace(' ', '_')
            url = f"{self.base_url}/team/{team_slug}/{season}"
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            stats = {
                'team_name': team_name,
                'season': season,
                'xg_for_total': 0.0,
                'xg_against_total': 0.0,
                'xg_for_per_match': 0.0,
                'xg_against_per_match': 0.0,
                'shots_total': 0,
                'shots_on_target': 0,
                'data_source': 'understat',
                'last_updated': pd.Timestamp.now()
            }
            
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'teamsData' in script.string:
                    try:
                        json_data = script.string
                        start_idx = json_data.find('JSON.parse(\'') + 12
                        end_idx = json_data.find('\')', start_idx)
                        json_str = json_data[start_idx:end_idx].encode().decode('unicode_escape')
                        data = json.loads(json_str)
                        
                        if team_slug in data:
                            team_data = data[team_slug]
                            stats['xg_for_total'] = float(team_data.get('xG', 0))
                            stats['xg_against_total'] = float(team_data.get('xGA', 0))
                            stats['shots_total'] = int(team_data.get('shots', 0))
                            
                            matches_played = int(team_data.get('matches', 1))
                            if matches_played > 0:
                                stats['xg_for_per_match'] = stats['xg_for_total'] / matches_played
                                stats['xg_against_per_match'] = stats['xg_against_total'] / matches_played
                        
                        break
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Error parsing JSON for {team_name}: {e}")
            
            logger.info(f"Successfully scraped Understat xG data for {team_name}")
            return stats
            
        except Exception as e:
            logger.error(f"Error scraping Understat for {team_name}: {e}")
            return {
                'team_name': team_name,
                'season': season,
                'xg_for_total': None,
                'xg_against_total': None,
                'xg_for_per_match': None,
                'xg_against_per_match': None,
                'shots_total': None,
                'shots_on_target': None,
                'data_source': 'understat',
                'last_updated': pd.Timestamp.now(),
                'error': str(e)
            }
    
    def get_match_xg(self, match_id: str) -> Dict:
        """
        Scrape detailed xG data for a specific match.
        
        Args:
            match_id: Understat match ID
            
        Returns:
            Dictionary containing match xG details
        """
        try:
            time.sleep(2)
            url = f"{self.base_url}/match/{match_id}"
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            match_data = {
                'match_id': match_id,
                'home_xg': 0.0,
                'away_xg': 0.0,
                'home_shots': 0,
                'away_shots': 0,
                'data_source': 'understat',
                'last_updated': pd.Timestamp.now()
            }
            
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'rostersData' in script.string:
                    try:
                        json_data = script.string
                        start_idx = json_data.find('JSON.parse(\'') + 12
                        end_idx = json_data.find('\')', start_idx)
                        json_str = json_data[start_idx:end_idx].encode().decode('unicode_escape')
                        data = json.loads(json_str)
                        
                        if 'h' in data and 'a' in data:
                            home_xg = sum([float(p.get('xG', 0)) for p in data['h'].values()])
                            away_xg = sum([float(p.get('xG', 0)) for p in data['a'].values()])
                            
                            match_data['home_xg'] = home_xg
                            match_data['away_xg'] = away_xg
                            match_data['home_shots'] = len(data['h'])
                            match_data['away_shots'] = len(data['a'])
                        
                        break
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Error parsing match data: {e}")
            
            logger.info(f"Successfully scraped match xG for match {match_id}")
            return match_data
            
        except Exception as e:
            logger.error(f"Error scraping match {match_id}: {e}")
            return {
                'match_id': match_id,
                'home_xg': None,
                'away_xg': None,
                'home_shots': None,
                'away_shots': None,
                'data_source': 'understat',
                'last_updated': pd.Timestamp.now(),
                'error': str(e)
            }
