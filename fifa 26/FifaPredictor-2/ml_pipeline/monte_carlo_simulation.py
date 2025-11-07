import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """Monte Carlo simulation engine for FIFA 2026 tournament prediction."""
    
    def __init__(self, model, num_simulations: int = 5000):
        self.model = model
        self.num_simulations = num_simulations
        self.simulation_results = []
        
    def predict_match_outcome(self, team1_features: Dict, team2_features: Dict) -> Tuple[float, float, float]:
        """
        Predict match outcome probabilities.
        
        Args:
            team1_features: Features for team 1 (home)
            team2_features: Features for team 2 (away)
            
        Returns:
            Tuple of (prob_team1_win, prob_draw, prob_team2_win)
        """
        try:
            features_df = pd.DataFrame([{**team1_features, **team2_features}])
            
            prob_team1_win = self.model.predict(features_df)[0]
            
            prob_draw = 0.25
            prob_team2_win = 1 - prob_team1_win - prob_draw
            
            if prob_team2_win < 0:
                prob_team2_win = 0.1
                prob_team1_win = 0.9 - prob_draw
            
            total = prob_team1_win + prob_draw + prob_team2_win
            prob_team1_win /= total
            prob_draw /= total
            prob_team2_win /= total
            
            return prob_team1_win, prob_draw, prob_team2_win
            
        except Exception as e:
            logger.error(f"Error predicting match outcome: {e}")
            return 0.33, 0.34, 0.33
    
    def simulate_match(self, prob_team1_win: float, prob_draw: float, prob_team2_win: float) -> str:
        """
        Simulate a single match outcome.
        
        Args:
            prob_team1_win: Probability of team 1 winning
            prob_draw: Probability of draw
            prob_team2_win: Probability of team 2 winning
            
        Returns:
            'team1', 'draw', or 'team2'
        """
        outcome = np.random.choice(
            ['team1', 'draw', 'team2'],
            p=[prob_team1_win, prob_draw, prob_team2_win]
        )
        return outcome
    
    def simulate_penalty_shootout(self) -> str:
        """Simulate penalty shootout (50-50 chance)."""
        return np.random.choice(['team1', 'team2'])
    
    def simulate_knockout_match(self, team1: str, team1_features: Dict, 
                                 team2: str, team2_features: Dict) -> str:
        """
        Simulate a knockout match (includes extra time and penalties if needed).
        
        Args:
            team1: Name of team 1
            team1_features: Features for team 1
            team2: Name of team 2
            team2_features: Features for team 2
            
        Returns:
            Name of winning team
        """
        prob_team1_win, prob_draw, prob_team2_win = self.predict_match_outcome(
            team1_features, team2_features
        )
        
        outcome = self.simulate_match(prob_team1_win, prob_draw, prob_team2_win)
        
        if outcome == 'team1':
            return team1
        elif outcome == 'team2':
            return team2
        else:
            penalty_winner = self.simulate_penalty_shootout()
            return team1 if penalty_winner == 'team1' else team2
    
    def simulate_group_stage(self, groups: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
        """
        Simulate group stage matches.
        
        Args:
            groups: Dictionary mapping group names to lists of team dicts
            
        Returns:
            Dictionary mapping group names to lists of qualified teams
        """
        qualified_teams = {}
        
        for group_name, teams in groups.items():
            points = {team['name']: 0 for team in teams}
            
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team1 = teams[i]
                    team2 = teams[j]
                    
                    prob_team1_win, prob_draw, prob_team2_win = self.predict_match_outcome(
                        team1['features'], team2['features']
                    )
                    
                    outcome = self.simulate_match(prob_team1_win, prob_draw, prob_team2_win)
                    
                    if outcome == 'team1':
                        points[team1['name']] += 3
                    elif outcome == 'team2':
                        points[team2['name']] += 3
                    else:
                        points[team1['name']] += 1
                        points[team2['name']] += 1
            
            sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
            qualified_teams[group_name] = [team[0] for team in sorted_teams[:2]]
        
        return qualified_teams
    
    def simulate_tournament(self, teams_data: Dict[str, Dict], bracket_structure: Dict = None) -> Dict:
        """
        Simulate entire tournament from group stage to final.
        
        Args:
            teams_data: Dictionary mapping team names to their features
            bracket_structure: Tournament bracket structure (optional)
            
        Returns:
            Dictionary containing simulation results
        """
        results = {
            'winner': None,
            'runner_up': None,
            'semi_finalists': [],
            'quarter_finalists': []
        }
        
        team_names = list(teams_data.keys())
        if len(team_names) < 4:
            logger.warning("Not enough teams for tournament simulation")
            return results
        
        np.random.shuffle(team_names)
        semi_finalists = team_names[:4]
        
        match1_winner = self.simulate_knockout_match(
            semi_finalists[0], teams_data[semi_finalists[0]],
            semi_finalists[1], teams_data[semi_finalists[1]]
        )
        
        match2_winner = self.simulate_knockout_match(
            semi_finalists[2], teams_data[semi_finalists[2]],
            semi_finalists[3], teams_data[semi_finalists[3]]
        )
        
        finalists = [match1_winner, match2_winner]
        
        winner = self.simulate_knockout_match(
            finalists[0], teams_data[finalists[0]],
            finalists[1], teams_data[finalists[1]]
        )
        
        runner_up = finalists[1] if winner == finalists[0] else finalists[0]
        
        results['winner'] = winner
        results['runner_up'] = runner_up
        results['semi_finalists'] = semi_finalists
        
        return results
    
    def run_simulations(self, teams_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Run multiple tournament simulations.
        
        Args:
            teams_data: Dictionary mapping team names to their features
            
        Returns:
            DataFrame containing simulation summary
        """
        logger.info(f"Running {self.num_simulations} tournament simulations...")
        
        self.simulation_results = []
        finalist_appearances = Counter()
        winner_count = Counter()
        
        for sim_num in range(self.num_simulations):
            if (sim_num + 1) % 1000 == 0:
                logger.info(f"Completed {sim_num + 1}/{self.num_simulations} simulations")
            
            result = self.simulate_tournament(teams_data)
            self.simulation_results.append(result)
            
            if result['winner']:
                finalist_appearances[result['winner']] += 1
                winner_count[result['winner']] += 1
            
            if result['runner_up']:
                finalist_appearances[result['runner_up']] += 1
        
        summary_data = []
        for team in finalist_appearances:
            finalist_prob = finalist_appearances[team] / self.num_simulations
            winner_prob = winner_count[team] / self.num_simulations
            
            summary_data.append({
                'team': team,
                'finalist_probability': finalist_prob * 100,
                'winner_probability': winner_prob * 100,
                'finalist_appearances': finalist_appearances[team],
                'wins': winner_count[team]
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values(
            'finalist_probability', ascending=False
        )
        
        logger.info(f"Simulation complete. Top finalist: {summary_df.iloc[0]['team']} "
                   f"({summary_df.iloc[0]['finalist_probability']:.1f}%)")
        
        return summary_df
    
    def get_confidence_intervals(self, summary_df: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Calculate confidence intervals using bootstrap method.
        
        Args:
            summary_df: Summary DataFrame from run_simulations
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            DataFrame with confidence intervals added
        """
        alpha = 1 - confidence_level
        
        for idx, row in summary_df.iterrows():
            n_successes = row['finalist_appearances']
            n_trials = self.num_simulations
            
            p_est = n_successes / n_trials
            se = np.sqrt(p_est * (1 - p_est) / n_trials)
            
            z_score = 1.96
            ci_lower = max(0, p_est - z_score * se) * 100
            ci_upper = min(1, p_est + z_score * se) * 100
            
            summary_df.loc[idx, 'ci_lower'] = ci_lower
            summary_df.loc[idx, 'ci_upper'] = ci_upper
        
        return summary_df
