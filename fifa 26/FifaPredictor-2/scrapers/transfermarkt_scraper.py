import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import Dict, List, Optional
from requests_cache import CachedSession
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransfermarktScraper:
    """Scrapes squad market values, player ages, and transfer data from Transfermarkt."""
    
    def __init__(self, cache_expire_after=7200):
        self.base_url = "https://www.transfermarkt.com"
        self.session = CachedSession('transfermarkt_cache', expire_after=cache_expire_after)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def parse_market_value(self, value_str: str) -> float:
        """Convert market value string to millions of euros."""
        try:
            if not value_str or value_str == '-':
                return 0.0
            
            value_str = value_str.replace('€', '').replace('m', '').replace('k', '').strip()
            
            if 'm' in value_str.lower():
                return float(re.sub(r'[^\d.]', '', value_str))
            elif 'k' in value_str.lower():
                return float(re.sub(r'[^\d.]', '', value_str)) / 1000
            else:
                try:
                    return float(value_str)
                except:
                    return 0.0
        except:
            return 0.0
    
    def get_squad_market_value(self, team_name: str, team_id: Optional[str] = None) -> Dict:
        """
        Scrape squad market value and player demographics.
        
        Args:
            team_name: Name of the team
            team_id: Transfermarkt team ID (optional)
            
        Returns:
            Dictionary containing squad value and demographics
        """
        try:
            time.sleep(3)
            
            if not team_id:
                search_url = f"{self.base_url}/schnellsuche/ergebnis/schnellsuche?query={team_name.replace(' ', '+')}"
                response = self.session.get(search_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'lxml')
                
                team_link = soup.find('a', {'class': 'vereinprofil_tooltip'})
                if team_link and 'href' in team_link.attrs:
                    team_id = team_link['href'].split('/')[-1]
                else:
                    logger.warning(f"Could not find team ID for {team_name}")
                    return self._empty_squad_data(team_name)
            
            squad_url = f"{self.base_url}/team/kader/verein/{team_id}"
            response = self.session.get(squad_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            stats = {
                'team_name': team_name,
                'squad_value_million_eur': 0.0,
                'avg_age': 0.0,
                'total_players': 0,
                'foreigners_count': 0,
                'avg_market_value_per_player': 0.0,
                'data_source': 'transfermarkt',
                'last_updated': pd.Timestamp.now()
            }
            
            market_value_elem = soup.find('a', {'class': 'data-header__market-value-wrapper'})
            if market_value_elem:
                value_text = market_value_elem.text.strip()
                stats['squad_value_million_eur'] = self.parse_market_value(value_text)
            
            info_table = soup.find('div', {'class': 'large-5 columns'})
            if info_table:
                avg_age_span = info_table.find('span', string=re.compile('Average age:'))
                if avg_age_span and avg_age_span.parent:
                    try:
                        age_text = avg_age_span.parent.text.split(':')[-1].strip()
                        stats['avg_age'] = float(age_text)
                    except (ValueError, IndexError):
                        pass
                
                squad_size_span = info_table.find('span', string=re.compile('Squad size:'))
                if squad_size_span and squad_size_span.parent:
                    try:
                        size_text = squad_size_span.parent.text.split(':')[-1].strip()
                        stats['total_players'] = int(size_text)
                    except (ValueError, IndexError):
                        pass
                
                foreigners_span = info_table.find('span', string=re.compile('Foreigners:'))
                if foreigners_span and foreigners_span.parent:
                    try:
                        foreigners_text = foreigners_span.parent.text.split(':')[-1].strip().split()[0]
                        stats['foreigners_count'] = int(foreigners_text)
                    except (ValueError, IndexError):
                        pass
            
            if stats['total_players'] > 0 and stats['squad_value_million_eur'] > 0:
                stats['avg_market_value_per_player'] = stats['squad_value_million_eur'] / stats['total_players']
            
            logger.info(f"Successfully scraped Transfermarkt data for {team_name}")
            return stats
            
        except Exception as e:
            logger.error(f"Error scraping Transfermarkt for {team_name}: {e}")
            return self._empty_squad_data(team_name, error=str(e))
    
    def _empty_squad_data(self, team_name: str, error: str = None) -> Dict:
        """Return empty squad data structure."""
        data = {
            'team_name': team_name,
            'squad_value_million_eur': None,
            'avg_age': None,
            'total_players': None,
            'foreigners_count': None,
            'avg_market_value_per_player': None,
            'data_source': 'transfermarkt',
            'last_updated': pd.Timestamp.now()
        }
        if error:
            data['error'] = error
        return data
    
    def get_player_list(self, team_id: str) -> List[Dict]:
        """
        Get list of all players with their market values.
        
        Args:
            team_id: Transfermarkt team ID
            
        Returns:
            List of dictionaries containing player data
        """
        try:
            time.sleep(3)
            
            squad_url = f"{self.base_url}/team/kader/verein/{team_id}"
            response = self.session.get(squad_url, timeout=15)
            response.raise_for_status()
            
            tables = pd.read_html(response.content)
            
            if tables:
                df = tables[0]
                players = df.to_dict('records')
                logger.info(f"Successfully scraped {len(players)} players for team {team_id}")
                return players
            
            return []
            
        except Exception as e:
            logger.error(f"Error scraping player list for team {team_id}: {e}")
            return []
