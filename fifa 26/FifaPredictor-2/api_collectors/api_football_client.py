import requests
import pandas as pd
import time
import logging
from typing import Dict, List, Optional
from requests_cache import CachedSession
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIFootballClient:
    """Client for API-Football (api-sports.io) with rate limiting and caching."""
    
    def __init__(self, api_key: Optional[str] = None, cache_expire_after=600):
        self.api_key = api_key or os.getenv('API_FOOTBALL_KEY', '')
        self.base_url = "https://v3.football.api-sports.io"
        self.session = CachedSession('api_football_cache', expire_after=cache_expire_after)
        self.session.headers.update({
            'x-apisports-key': self.api_key
        })
        self.rate_limit_delay = 1
        self.requests_made = 0
        self.max_requests_per_day = 100
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting."""
        if self.requests_made >= self.max_requests_per_day:
            logger.warning("Daily API request limit reached")
            return {'response': []}
        
        try:
            time.sleep(self.rate_limit_delay)
            
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            self.requests_made += 1
            data = response.json()
            
            logger.info(f"API request successful: {endpoint} (requests: {self.requests_made}/{self.max_requests_per_day})")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            return {'response': [], 'error': str(e)}
    
    def get_fixtures(self, league_id: int, season: int) -> List[Dict]:
        """
        Get fixtures for a specific league and season.
        
        Args:
            league_id: API-Football league ID (e.g., 1 for World Cup)
            season: Year of the season
            
        Returns:
            List of fixture dictionaries
        """
        endpoint = "fixtures"
        params = {
            'league': league_id,
            'season': season
        }
        
        data = self._make_request(endpoint, params)
        fixtures = data.get('response', [])
        
        logger.info(f"Retrieved {len(fixtures)} fixtures for league {league_id}, season {season}")
        return fixtures
    
    def get_team_statistics(self, team_id: int, league_id: int, season: int) -> Dict:
        """
        Get team statistics for a specific league and season.
        
        Args:
            team_id: API-Football team ID
            league_id: API-Football league ID
            season: Year of the season
            
        Returns:
            Dictionary containing team statistics
        """
        endpoint = "teams/statistics"
        params = {
            'team': team_id,
            'league': league_id,
            'season': season
        }
        
        data = self._make_request(endpoint, params)
        
        if data.get('response'):
            stats = data['response']
            logger.info(f"Retrieved statistics for team {team_id}")
            return stats
        
        return {}
    
    def get_teams(self, league_id: int, season: int) -> List[Dict]:
        """
        Get all teams in a league for a specific season.
        
        Args:
            league_id: API-Football league ID
            season: Year of the season
            
        Returns:
            List of team dictionaries
        """
        endpoint = "teams"
        params = {
            'league': league_id,
            'season': season
        }
        
        data = self._make_request(endpoint, params)
        teams = data.get('response', [])
        
        logger.info(f"Retrieved {len(teams)} teams for league {league_id}, season {season}")
        return teams
    
    def get_standings(self, league_id: int, season: int) -> List[Dict]:
        """
        Get league standings.
        
        Args:
            league_id: API-Football league ID
            season: Year of the season
            
        Returns:
            List of standings data
        """
        endpoint = "standings"
        params = {
            'league': league_id,
            'season': season
        }
        
        data = self._make_request(endpoint, params)
        
        if data.get('response'):
            standings = data['response'][0].get('league', {}).get('standings', [[]])[0]
            logger.info(f"Retrieved standings for league {league_id}")
            return standings
        
        return []
    
    def get_h2h(self, team1_id: int, team2_id: int, last_n: int = 10) -> List[Dict]:
        """
        Get head-to-head matches between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            last_n: Number of recent matches to retrieve
            
        Returns:
            List of match dictionaries
        """
        endpoint = "fixtures/headtohead"
        params = {
            'h2h': f"{team1_id}-{team2_id}",
            'last': last_n
        }
        
        data = self._make_request(endpoint, params)
        matches = data.get('response', [])
        
        logger.info(f"Retrieved {len(matches)} H2H matches")
        return matches
    
    def get_api_status(self) -> Dict:
        """Get API usage statistics."""
        return {
            'requests_made': self.requests_made,
            'max_requests': self.max_requests_per_day,
            'remaining': self.max_requests_per_day - self.requests_made,
            'percentage_used': (self.requests_made / self.max_requests_per_day) * 100
        }
