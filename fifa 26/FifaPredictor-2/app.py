import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from app.components.home_tab import render_home_tab
from app.components.fixtures_tab import render_fixtures_tab
from app.components.compare_tab import render_compare_tab
from app.components.stats_tab import render_stats_tab
from app.components.evaluation_tab import render_evaluation_tab
from app.data_loader import DataLoader

st.set_page_config(
    page_title="FIFA 2026 Finalist Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_data_loader():
    """Get singleton data loader instance."""
    return DataLoader()

@st.cache_resource
def load_teams_data():
    """Load teams data from precomputed ETL outputs or use sample data."""
    loader = get_data_loader()
    
    if loader.is_data_available():
        try:
            teams_data = loader.load_teams_data()
            if teams_data:
                return teams_data
        except Exception as e:
            st.warning(f"⚠️ Could not load real data: {e}. Using sample data instead.")
    
    return load_sample_data()

@st.cache_resource
def load_sample_data():
    """Load sample data for demonstration."""
    
    fifa_teams_48 = [
        'Brazil', 'Argentina', 'France', 'Germany', 'Spain', 'England', 'Belgium', 'Netherlands',
        'Portugal', 'Italy', 'Croatia', 'Uruguay', 'Colombia', 'Mexico', 'USA', 'Switzerland',
        'Denmark', 'Senegal', 'Morocco', 'Japan', 'South Korea', 'Iran', 'Australia', 'Poland',
        'Sweden', 'Austria', 'Ukraine', 'Turkey', 'Czech Republic', 'Wales', 'Serbia', 'Russia',
        'Peru', 'Chile', 'Nigeria', 'Egypt', 'Ghana', 'Cameroon', 'Tunisia', 'Algeria',
        'Ecuador', 'Paraguay', 'Canada', 'Costa Rica', 'Jamaica', 'Saudi Arabia', 'Qatar', 'Iraq'
    ]
    
    teams_data = {}
    for i, team in enumerate(fifa_teams_48):
        teams_data[team] = {
            'fifa_rank': i + 1,
            'elo_rating': 2000 - (i * 20) + np.random.randint(-50, 50),
            'xg_for_90': max(0.5, 2.5 - (i * 0.03) + np.random.uniform(-0.3, 0.3)),
            'xg_against_90': max(0.3, 0.8 + (i * 0.02) + np.random.uniform(-0.2, 0.2)),
            'shots_per_90': max(8, 18 - (i * 0.1) + np.random.uniform(-2, 2)),
            'sot_per_90': max(4, 8 - (i * 0.05) + np.random.uniform(-1, 1)),
            'possession_pct': max(40, 58 - (i * 0.2) + np.random.uniform(-3, 3)),
            'pass_accuracy': max(70, 85 - (i * 0.1) + np.random.uniform(-2, 2)),
            'squad_value_million_eur': max(50, 1200 - (i * 15) + np.random.uniform(-50, 50)),
            'avg_age': 26 + np.random.uniform(-2, 2),
            'total_players': 23,
            'foreigners_count': np.random.randint(0, 10),
            'clean_sheet_pct': max(20, 50 - (i * 0.3) + np.random.uniform(-5, 5)),
            'avg_goals_for_90': max(0.5, 2.2 - (i * 0.02) + np.random.uniform(-0.3, 0.3)),
            'avg_goals_against_90': max(0.3, 0.9 + (i * 0.02) + np.random.uniform(-0.2, 0.2)),
            'progressive_passes_per_90': max(30, 80 - (i * 0.5) + np.random.uniform(-5, 5)),
            'passes_per_90': max(300, 600 - (i * 3) + np.random.uniform(-20, 20)),
            'injuries_count': np.random.randint(0, 5),
            'confederation': 'UEFA' if i < 20 else ('CONMEBOL' if i < 30 else ('CAF' if i < 38 else 'CONCACAF')),
            'coach_name': f'Coach {team}',
            'rank_delta': np.random.randint(-5, 5)
        }
    
    return teams_data

@st.cache_resource
def load_simulation_results():
    """Load precomputed simulation results or generate sample."""
    loader = get_data_loader()
    
    if loader.is_data_available():
        try:
            results = loader.load_simulation_results()
            if not results.empty:
                return results
        except Exception as e:
            st.warning(f"Using sample simulation: {e}")
    
    teams_data = load_sample_data()
    return generate_sample_simulation(teams_data)

@st.cache_resource
def generate_sample_simulation(teams_data):
    """Generate sample simulation results (fallback)."""
    team_names = list(teams_data.keys())
    
    results = []
    for i, team in enumerate(team_names):
        base_prob = max(0.5, 25 - (i * 0.4) + np.random.uniform(-2, 2))
        
        results.append({
            'team': team,
            'finalist_probability': base_prob,
            'winner_probability': base_prob * 0.5,
            'finalist_appearances': int(base_prob * 50),
            'wins': int(base_prob * 25),
            'ci_lower': max(0, base_prob - 2),
            'ci_upper': min(100, base_prob + 2),
            'top_features': [
                'Squad Value',
                'FIFA Rank',
                'xG Differential'
            ]
        })
    
    df = pd.DataFrame(results).sort_values('finalist_probability', ascending=False)
    return df

@st.cache_resource
def load_fixtures_data():
    """Load precomputed fixtures or generate sample."""
    loader = get_data_loader()
    
    if loader.is_data_available():
        try:
            fixtures = loader.load_fixtures()
            if not fixtures.empty:
                return fixtures
        except Exception as e:
            st.warning(f"Using sample fixtures: {e}")
    
    return generate_sample_fixtures()

@st.cache_resource
def generate_sample_fixtures():
    """Generate sample fixtures data (fallback)."""
    
    fixtures = []
    teams = ['Brazil', 'Argentina', 'France', 'Germany', 'Spain', 'England']
    
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

@st.cache_resource
def generate_model_metrics():
    """Generate sample model metrics."""
    
    return {
        'roc_auc': 0.872,
        'pr_auc': 0.765,
        'brier_score': 0.142,
        'log_loss': 0.389,
        'accuracy': 0.823,
        'precision': 0.781,
        'recall': 0.745,
        'f1_score': 0.763,
        'confusion_matrix': [[380, 45], [62, 213]],
        'threshold': 0.5,
        'n_samples': 700,
        'n_positive': 275,
        'n_negative': 425,
        'training_date': '2025-01-07',
        'roc_curve': {
            'fpr': list(np.linspace(0, 1, 50)),
            'tpr': list(np.linspace(0, 1, 50) ** 0.5),
            'thresholds': list(np.linspace(1, 0, 50))
        },
        'pr_curve': {
            'precision': list(1 - np.linspace(0, 1, 50) * 0.3),
            'recall': list(np.linspace(0, 1, 50)),
            'thresholds': list(np.linspace(1, 0, 50))
        }
    }

@st.cache_resource
def generate_feature_importance():
    """Generate sample feature importance data."""
    
    features = [
        'squad_value_million_eur', 'fifa_rank', 'xg_for_90', 'elo_rating',
        'xg_against_90', 'win_rate_last5', 'possession_pct', 'pass_accuracy',
        'avg_goals_for_90', 'shots_per_90', 'clean_sheet_pct', 'avg_age',
        'progressive_passes_per_90', 'sot_per_90', 'avg_goals_against_90',
        'implied_prob_win', 'xg_differential', 'attack_minus_defense',
        'foreigners_count', 'total_players'
    ]
    
    importance_values = [850, 720, 680, 650, 590, 560, 520, 480, 450, 420,
                        390, 360, 330, 310, 290, 270, 250, 230, 210, 190]
    
    return pd.DataFrame({
        'feature': features,
        'importance': importance_values
    })

def main():
    """Main application."""
    
    st.sidebar.title("⚽ FIFA 2026 Predictor")
    st.sidebar.markdown("---")
    
    tab_selection = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📅 Fixtures", "⚔️ Compare", "📊 Stats", "🎯 Evaluation"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Data Management")
    
    loader = get_data_loader()
    metadata = loader.load_metadata()
    
    data_freshness = loader.get_data_freshness()
    st.sidebar.caption(f"📊 Data Status: {data_freshness}")
    
    if st.sidebar.button("🔄 Refresh Real Data"):
        with st.spinner("Collecting real data from web scrapers... This may take several minutes."):
            result = loader.trigger_refresh()
            st.sidebar.success(result)
            st.cache_resource.clear()
            st.rerun()
    
    if st.sidebar.button("♻️ Clear Cache"):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Data Sources")
    st.sidebar.caption("• FBref (xG, shots, possession)")
    st.sidebar.caption("• Transfermarkt (squad values)")
    st.sidebar.caption("• API-Football (fixtures)")
    st.sidebar.caption("• football-data.co.uk (historical)")
    
    st.sidebar.markdown("---")
    num_teams = metadata.get('num_teams', 48)
    st.sidebar.caption(f"Teams: {num_teams}")
    st.sidebar.caption(f"ETL Version: {metadata.get('etl_version', 'N/A')}")
    
    teams_data = load_teams_data()
    simulation_results = load_simulation_results()
    fixtures_data = load_fixtures_data()
    model_metrics = generate_model_metrics()
    feature_importance = generate_feature_importance()
    
    model_info = {
        'last_updated': '2025-01-07 12:00:00',
        'num_simulations': 5000
    }
    
    api_status = {
        'requests_made': 45,
        'max_requests': 100,
        'remaining': 55,
        'percentage_used': 45
    }
    
    if tab_selection == "🏠 Home":
        render_home_tab(simulation_results, model_info)
    
    elif tab_selection == "📅 Fixtures":
        render_fixtures_tab(fixtures_data, api_status)
    
    elif tab_selection == "⚔️ Compare":
        render_compare_tab(teams_data, None)
    
    elif tab_selection == "📊 Stats":
        render_stats_tab(teams_data)
    
    elif tab_selection == "🎯 Evaluation":
        render_evaluation_tab(model_metrics, feature_importance)
    
    st.markdown("---")
    st.caption("FIFA 2026 Finalist Predictor | Built with LightGBM, Streamlit, and Real-Time Data")

if __name__ == "__main__":
    main()
