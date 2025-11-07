import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List

def render_home_tab(simulation_results: pd.DataFrame, model_info: Dict):
    """Render the Home tab with finalist predictions and top 10 teams."""
    
    st.title("FIFA 2026 Finalist Predictor")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "LightGBM")
    with col2:
        st.metric("Last Updated", model_info.get('last_updated', 'N/A'))
    with col3:
        st.metric("Simulations", f"{model_info.get('num_simulations', 5000):,}")
    
    st.markdown("---")
    
    st.header("🏆 Predicted Finalists")
    
    if len(simulation_results) >= 2:
        top_2 = simulation_results.head(2)
        
        col1, col2 = st.columns(2)
        
        for idx, (i, team) in enumerate(top_2.iterrows()):
            with col1 if idx == 0 else col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 15px; color: white;'>
                    <h2 style='margin: 0;'>{'🥇' if idx == 0 else '🥈'} Finalist #{idx + 1}</h2>
                    <h1 style='margin: 10px 0; font-size: 2.5em;'>{team['team']}</h1>
                    <h2 style='margin: 0;'>{team['finalist_probability']:.1f}%</h2>
                    <p style='margin: 5px 0; opacity: 0.9;'>
                        95% CI: [{team.get('ci_lower', 0):.1f}% - {team.get('ci_upper', 0):.1f}%]
                    </p>
                    <p style='margin: 5px 0; opacity: 0.8; font-size: 0.9em;'>
                        Finalist in {team['finalist_appearances']:,} / {model_info.get('num_simulations', 5000):,} simulations
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if 'top_features' in team and team['top_features']:
                    st.caption("Top Contributing Factors:")
                    for feature in team.get('top_features', [])[:3]:
                        st.caption(f"• {feature}")
    
    st.markdown("---")
    
    st.header("📊 Top 10 Teams")
    
    if len(simulation_results) >= 10:
        top_10 = simulation_results.head(10).copy()
        
        top_10_display = top_10[[
            'team', 'finalist_probability', 'winner_probability',
            'finalist_appearances', 'wins'
        ]].copy()
        
        top_10_display.columns = [
            'Team', 'Finalist %', 'Winner %',
            'Finalist Appearances', 'Wins'
        ]
        
        top_10_display['Finalist %'] = top_10_display['Finalist %'].round(2)
        top_10_display['Winner %'] = top_10_display['Winner %'].round(2)
        
        st.dataframe(
            top_10_display,
            width='stretch',
            hide_index=True
        )
    
    st.markdown("---")
    
    st.header("📈 Simulation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simulation Settings")
        num_sims = st.number_input(
            "Number of Simulations",
            min_value=1000,
            max_value=20000,
            value=model_info.get('num_simulations', 5000),
            step=1000
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=99999,
            value=42
        )
        
        if st.button("Run New Simulation", type="primary"):
            st.info("Simulation feature coming soon!")
    
    with col2:
        st.subheader("Top 20 Teams - Finalist Probability")
        
        if len(simulation_results) >= 20:
            top_20 = simulation_results.head(20)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_20['finalist_probability'],
                    y=top_20['team'],
                    orientation='h',
                    marker=dict(
                        color=top_20['finalist_probability'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Probability %")
                    ),
                    text=top_20['finalist_probability'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Finalist Probability by Team",
                xaxis_title="Probability (%)",
                yaxis_title="Team",
                height=600,
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = simulation_results.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="fifa_2026_predictions.csv",
            mime="text/csv"
        )
    
    with col2:
        st.info("Click to download full simulation results")
