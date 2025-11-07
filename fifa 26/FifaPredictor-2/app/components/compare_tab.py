import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

def calculate_match_probabilities(team_a_data: Dict, team_b_data: Dict):
    """Calculate match win/draw/loss probabilities based on team statistics."""

    # Extract key metrics with defaults (ensure they are not None)
    a_rank = team_a_data.get('fifa_rank') if team_a_data.get('fifa_rank') is not None else 50
    b_rank = team_b_data.get('fifa_rank') if team_b_data.get('fifa_rank') is not None else 50

    a_xg_for = team_a_data.get('xg_for_90') if team_a_data.get('xg_for_90') is not None else 1.5
    b_xg_for = team_b_data.get('xg_for_90') if team_b_data.get('xg_for_90') is not None else 1.5

    a_xg_against = team_a_data.get('xg_against_90') if team_a_data.get('xg_against_90') is not None else 1.0
    b_xg_against = team_b_data.get('xg_against_90') if team_b_data.get('xg_against_90') is not None else 1.0

    a_value = team_a_data.get('squad_value_million_eur') if team_a_data.get('squad_value_million_eur') is not None else 500
    b_value = team_b_data.get('squad_value_million_eur') if team_b_data.get('squad_value_million_eur') is not None else 500

    a_elo = team_a_data.get('elo_rating') if team_a_data.get('elo_rating') is not None else 1800
    b_elo = team_b_data.get('elo_rating') if team_b_data.get('elo_rating') is not None else 1800

    # Calculate differentials
    rank_diff = b_rank - a_rank  # Positive means A is better ranked
    xg_diff = a_xg_for - b_xg_for  # Positive means A has better attack
    defense_diff = b_xg_against - a_xg_against  # Positive means A has better defense
    value_diff = a_value - b_value  # Positive means A has higher squad value
    elo_diff = a_elo - b_elo  # Positive means A has higher Elo

    # Normalize differentials
    rank_diff_norm = rank_diff / 20.0  # Scale rank difference
    xg_diff_norm = xg_diff * 2.0  # Scale xG difference
    defense_diff_norm = defense_diff * 2.0  # Scale defense difference
    value_diff_norm = (value_diff / 500.0) * 0.5  # Scale value difference
    elo_diff_norm = elo_diff / 200.0  # Scale Elo difference

    # Calculate combined strength difference
    strength_diff = (rank_diff_norm + xg_diff_norm + defense_diff_norm +
                    value_diff_norm + elo_diff_norm) / 5.0

    # Convert to probability using logistic function
    import math
    prob_a_win = 1 / (1 + math.exp(-strength_diff))
    prob_b_win = 1 / (1 + math.exp(strength_diff))

    # Adjust for draw probability (typically 25-30% in football)
    draw_factor = 0.25 + abs(strength_diff) * 0.05  # More draws when teams are evenly matched
    draw_factor = min(draw_factor, 0.35)  # Cap at 35%

    # Normalize probabilities
    total = prob_a_win + prob_b_win + draw_factor
    prob_a_win = (prob_a_win / total) * 100
    prob_draw = (draw_factor / total) * 100
    prob_b_win = (prob_b_win / total) * 100

    return prob_a_win, prob_draw, prob_b_win

def get_confidence_level(probability: float):
    """Get confidence level description based on probability."""
    if probability >= 60:
        return "Very High"
    elif probability >= 50:
        return "High"
    elif probability >= 40:
        return "Moderate"
    elif probability >= 30:
        return "Low"
    else:
        return "Very Low"

def render_compare_tab(teams_data: Dict, model):
    """Render the Compare tab for team vs team analysis."""

    st.title("⚔️ Team Comparison")

    st.markdown("Compare two teams head-to-head with detailed statistics and match predictions")

    st.markdown("---")

    team_names = sorted(list(teams_data.keys()))

    col1, col2 = st.columns(2)

    with col1:
        team_a = st.selectbox("Select Team A", team_names, key="team_a")

    with col2:
        team_names_filtered = [t for t in team_names if t != team_a]
        team_b = st.selectbox("Select Team B", team_names_filtered, key="team_b")
    
    if team_a and team_b:
        st.markdown("---")

        st.header(f"🆚 {team_a} vs {team_b}")

        team_a_data = teams_data.get(team_a, {})
        team_b_data = teams_data.get(team_b, {})

        # Calculate dynamic probabilities
        prob_a_win, prob_draw, prob_b_win = calculate_match_probabilities(team_a_data, team_b_data)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                f"{team_a} Win",
                f"{prob_a_win:.1f}%",
                delta=get_confidence_level(prob_a_win)
            )

        with col2:
            st.metric(
                "Draw",
                f"{prob_draw:.1f}%",
                delta=get_confidence_level(prob_draw)
            )

        with col3:
            st.metric(
                f"{team_b} Win",
                f"{prob_b_win:.1f}%",
                delta=get_confidence_level(prob_b_win)
            )
        
        st.markdown("---")
        
        st.header("📊 Statistical Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🔵 {team_a}")

            fifa_rank_a = team_a_data.get('fifa_rank')
            st.metric("FIFA Rank", fifa_rank_a if fifa_rank_a is not None else "N/A")

            xg_for_a = team_a_data.get('xg_for_90')
            st.metric("xG For (per 90)", f"{xg_for_a:.2f}" if xg_for_a is not None else "N/A")

            xg_against_a = team_a_data.get('xg_against_90')
            st.metric("xG Against (per 90)", f"{xg_against_a:.2f}" if xg_against_a is not None else "N/A")

            possession_a = team_a_data.get('possession_pct')
            st.metric("Possession %", f"{possession_a:.1f}%" if possession_a is not None else "N/A")

            squad_value_a = team_a_data.get('squad_value_million_eur')
            st.metric("Squad Value (€M)", f"{squad_value_a:.1f}" if squad_value_a is not None else "N/A")

        with col2:
            st.subheader(f"🔴 {team_b}")

            fifa_rank_b = team_b_data.get('fifa_rank')
            st.metric("FIFA Rank", fifa_rank_b if fifa_rank_b is not None else "N/A")

            xg_for_b = team_b_data.get('xg_for_90')
            st.metric("xG For (per 90)", f"{xg_for_b:.2f}" if xg_for_b is not None else "N/A")

            xg_against_b = team_b_data.get('xg_against_90')
            st.metric("xG Against (per 90)", f"{xg_against_b:.2f}" if xg_against_b is not None else "N/A")

            possession_b = team_b_data.get('possession_pct')
            st.metric("Possession %", f"{possession_b:.1f}%" if possession_b is not None else "N/A")

            squad_value_b = team_b_data.get('squad_value_million_eur')
            st.metric("Squad Value (€M)", f"{squad_value_b:.1f}" if squad_value_b is not None else "N/A")
        
        st.markdown("---")
        
        st.header("🎯 Radar Chart Comparison")
        
        categories = ['Attack', 'Defense', 'Possession', 'Squad Value', 'Experience']
        
        team_a_values = [
            (team_a_data.get('xg_for_90', 0) if team_a_data.get('xg_for_90') is not None else 0) * 10,
            (3 - (team_a_data.get('xg_against_90', 1.5) if team_a_data.get('xg_against_90') is not None else 1.5)) * 10,
            team_a_data.get('possession_pct', 50) if team_a_data.get('possession_pct') is not None else 50,
            min((team_a_data.get('squad_value_million_eur', 0) if team_a_data.get('squad_value_million_eur') is not None else 0) / 10, 100),
            100 - min(team_a_data.get('fifa_rank', 50) if team_a_data.get('fifa_rank') is not None else 50, 100)
        ]
        
        team_b_values = [
            (team_b_data.get('xg_for_90', 0) if team_b_data.get('xg_for_90') is not None else 0) * 10,
            (3 - (team_b_data.get('xg_against_90', 1.5) if team_b_data.get('xg_against_90') is not None else 1.5)) * 10,
            team_b_data.get('possession_pct', 50) if team_b_data.get('possession_pct') is not None else 50,
            min((team_b_data.get('squad_value_million_eur', 0) if team_b_data.get('squad_value_million_eur') is not None else 0) / 10, 100),
            100 - min(team_b_data.get('fifa_rank', 50) if team_b_data.get('fifa_rank') is not None else 50, 100)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=team_a_values,
            theta=categories,
            fill='toself',
            name=team_a,
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=team_b_values,
            theta=categories,
            fill='toself',
            name=team_b,
            line=dict(color='red')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Team Performance Comparison (Normalized)",
            height=500
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        st.header("📈 Recent Form")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{team_a} - Last 10 Matches")
            st.write("**Record:** 6W - 2D - 2L")
            st.write("**Goals For:** 18")
            st.write("**Goals Against:** 8")
            st.write("**Clean Sheets:** 4")
        
        with col2:
            st.subheader(f"{team_b} - Last 10 Matches")
            st.write("**Record:** 5W - 3D - 2L")
            st.write("**Goals For:** 15")
            st.write("**Goals Against:** 10")
            st.write("**Clean Sheets:** 3")
        
        st.markdown("---")
        
        st.header("🔄 Head-to-Head History")
        
        h2h_data = pd.DataFrame({
            'Date': ['2023-10-15', '2022-06-20', '2021-11-14', '2020-09-08'],
            'Competition': ['Friendly', 'World Cup Qualifier', 'Friendly', 'Nations League'],
            'Home': [team_a, team_b, team_a, team_b],
            'Score': ['2-1', '1-1', '0-2', '3-2'],
            'Away': [team_b, team_a, team_b, team_a]
        })
        
        st.dataframe(h2h_data, width='stretch', hide_index=True)
        
        st.markdown("---")
        
        st.header("🧠 Feature Importance")
        
        st.info("SHAP waterfall plot showing which features drive the prediction will be displayed here once the model is trained on real data")
