import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineer 80+ features for FIFA 2026 finalist prediction."""
    
    def __init__(self):
        self.feature_names = []
    
    def calculate_elo_rating(self, team_matches: pd.DataFrame, k_factor: float = 32) -> float:
        """Calculate Elo rating based on match history."""
        elo = 1500
        
        for _, match in team_matches.iterrows():
            opponent_elo = match.get('opponent_elo', 1500)
            
            expected_score = 1 / (1 + 10 ** ((opponent_elo - elo) / 400))
            
            if match.get('result') == 'W':
                actual_score = 1
            elif match.get('result') == 'D':
                actual_score = 0.5
            else:
                actual_score = 0
            
            elo = elo + k_factor * (actual_score - expected_score)
        
        return elo
    
    def extract_form_features(self, team_matches: pd.DataFrame, n_matches: int = 5) -> Dict:
        """Extract recent form features."""
        recent = team_matches.head(n_matches)
        
        features = {
            f'win_rate_last{n_matches}': (recent['result'] == 'W').sum() / len(recent) if len(recent) > 0 else 0,
            f'points_last{n_matches}': ((recent['result'] == 'W').sum() * 3 + (recent['result'] == 'D').sum()) / len(recent) if len(recent) > 0 else 0,
            f'clean_sheet_pct_last{n_matches}': (recent['goals_against'] == 0).sum() / len(recent) if len(recent) > 0 else 0,
            f'avg_goals_for_last{n_matches}': recent['goals_for'].mean() if len(recent) > 0 else 0,
            f'avg_goals_against_last{n_matches}': recent['goals_against'].mean() if len(recent) > 0 else 0,
        }
        
        return features
    
    def extract_rolling_features(self, team_matches: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> Dict:
        """Calculate rolling window features."""
        features = {}
        
        for window in windows:
            recent = team_matches.head(window)
            
            if len(recent) > 0:
                features[f'avg_goals_for_90_last{window}'] = (recent['goals_for'].sum() / len(recent)) if len(recent) > 0 else 0
                features[f'avg_goals_against_90_last{window}'] = (recent['goals_against'].sum() / len(recent)) if len(recent) > 0 else 0
                features[f'win_rate_last{window}'] = (recent['result'] == 'W').sum() / len(recent)
                features[f'points_per_game_last{window}'] = ((recent['result'] == 'W').sum() * 3 + (recent['result'] == 'D').sum()) / len(recent)
        
        return features
    
    def extract_xg_features(self, team_stats: Dict) -> Dict:
        """Extract expected goals (xG) features."""
        features = {
            'xg_for_90': team_stats.get('xg_for_90', 0),
            'xg_against_90': team_stats.get('xg_against_90', 0),
            'xg_differential': team_stats.get('xg_for_90', 0) - team_stats.get('xg_against_90', 0),
            'xg_for_per_shot': team_stats.get('xg_for_90', 0) / max(team_stats.get('shots_per_90', 1), 1),
        }
        
        return features
    
    def extract_shooting_features(self, team_stats: Dict) -> Dict:
        """Extract shooting and chance creation features."""
        features = {
            'shots_per_90': team_stats.get('shots_per_90', 0),
            'sot_per_90': team_stats.get('sot_per_90', 0),
            'shot_accuracy': team_stats.get('sot_per_90', 0) / max(team_stats.get('shots_per_90', 1), 1) * 100,
            'conversion_rate': team_stats.get('goals_per_90', 0) / max(team_stats.get('shots_per_90', 1), 1) * 100,
        }
        
        return features
    
    def extract_possession_features(self, team_stats: Dict) -> Dict:
        """Extract possession and passing features."""
        features = {
            'possession_pct': team_stats.get('possession_pct', 50),
            'pass_accuracy': team_stats.get('pass_accuracy', 75),
            'passes_per_90': team_stats.get('passes_per_90', 400),
            'progressive_passes_per_90': team_stats.get('progressive_passes_per_90', 0),
        }
        
        return features
    
    def extract_squad_features(self, team_info: Dict) -> Dict:
        """Extract squad-level features."""
        features = {
            'squad_value_million_eur': team_info.get('squad_value_million_eur', 0),
            'avg_age': team_info.get('avg_age', 26),
            'total_players': team_info.get('total_players', 23),
            'foreigners_count': team_info.get('foreigners_count', 0),
            'avg_market_value_per_player': team_info.get('avg_market_value_per_player', 0),
        }
        
        return features
    
    def extract_betting_features(self, odds_data: Dict) -> Dict:
        """Extract market signals from betting odds."""
        features = {}
        
        if 'odds_home' in odds_data and odds_data['odds_home']:
            implied_prob_home = 1 / odds_data['odds_home']
            implied_prob_draw = 1 / odds_data.get('odds_draw', 3.5)
            implied_prob_away = 1 / odds_data.get('odds_away', 3.5)
            
            total = implied_prob_home + implied_prob_draw + implied_prob_away
            
            features['implied_prob_win'] = implied_prob_home / total
            features['implied_prob_draw'] = implied_prob_draw / total
            features['implied_prob_loss'] = implied_prob_away / total
            features['betting_favorite'] = 1 if implied_prob_home > implied_prob_away else 0
        
        return features
    
    def extract_contextual_features(self, team_info: Dict, match_info: Dict) -> Dict:
        """Extract contextual features like travel, climate, rest days."""
        features = {
            'confederation': team_info.get('confederation', 'UEFA'),
            'fifa_rank': team_info.get('fifa_rank', 50),
            'is_home_tournament': 1 if team_info.get('is_host', False) else 0,
            'days_rest': match_info.get('days_rest', 7),
            'travel_distance_km': match_info.get('travel_distance_km', 0),
            'climate_factor': match_info.get('climate_similar', 1),
        }
        
        return features
    
    def create_interaction_features(self, base_features: Dict) -> Dict:
        """Create interaction and derived features."""
        features = {}
        
        if 'avg_goals_for_90' in base_features and 'avg_goals_against_90' in base_features:
            features['attack_minus_defense'] = base_features['avg_goals_for_90'] - base_features['avg_goals_against_90']
        
        if 'xg_for_90' in base_features and 'xg_against_90' in base_features:
            features['xg_attack_minus_defense'] = base_features['xg_for_90'] - base_features['xg_against_90']
        
        if 'squad_value_million_eur' in base_features and 'fifa_rank' in base_features:
            features['value_rank_ratio'] = base_features['squad_value_million_eur'] / max(base_features['fifa_rank'], 1)
        
        return features
    
    def engineer_all_features(self, team_data: Dict, match_data: Dict = None) -> Dict:
        """
        Engineer all features for a team.
        
        Args:
            team_data: Dictionary containing team stats, squad info, match history
            match_data: Optional match-specific data (for match-level predictions)
            
        Returns:
            Dictionary containing all engineered features
        """
        all_features = {}
        
        if 'match_history' in team_data and len(team_data['match_history']) > 0:
            form_features = self.extract_form_features(team_data['match_history'])
            all_features.update(form_features)
            
            rolling_features = self.extract_rolling_features(team_data['match_history'])
            all_features.update(rolling_features)
            
            all_features['elo_rating'] = self.calculate_elo_rating(team_data['match_history'])
        
        if 'stats' in team_data:
            xg_features = self.extract_xg_features(team_data['stats'])
            all_features.update(xg_features)
            
            shooting_features = self.extract_shooting_features(team_data['stats'])
            all_features.update(shooting_features)
            
            possession_features = self.extract_possession_features(team_data['stats'])
            all_features.update(possession_features)
        
        if 'squad_info' in team_data:
            squad_features = self.extract_squad_features(team_data['squad_info'])
            all_features.update(squad_features)
        
        if 'team_info' in team_data:
            contextual_features = self.extract_contextual_features(team_data['team_info'], match_data or {})
            all_features.update(contextual_features)
        
        if match_data and 'odds' in match_data:
            betting_features = self.extract_betting_features(match_data['odds'])
            all_features.update(betting_features)
        
        interaction_features = self.create_interaction_features(all_features)
        all_features.update(interaction_features)
        
        self.feature_names = list(all_features.keys())
        logger.info(f"Engineered {len(all_features)} features")
        
        return all_features
    
    def get_feature_list(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names
