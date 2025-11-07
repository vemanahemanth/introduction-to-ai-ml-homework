import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict, List

def render_fixtures_tab(fixtures_df: pd.DataFrame, api_status: Dict):
    """Render the Fixtures tab with schedule and match probabilities."""
    
    st.title("📅 Fixtures & Schedule")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(fixtures_df))
    with col2:
        completed = len(fixtures_df[fixtures_df['status'] == 'finished'])
        st.metric("Completed", completed)
    with col3:
        upcoming = len(fixtures_df[fixtures_df['status'] == 'scheduled'])
        st.metric("Upcoming", upcoming)
    with col4:
        api_remaining = api_status.get('remaining', 0)
        st.metric("API Calls Remaining", api_remaining)
    
    st.markdown("---")
    
    st.header("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'date' in fixtures_df.columns:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now(), datetime.now())
            )
    
    with col2:
        if 'competition' in fixtures_df.columns:
            competitions = ['All'] + list(fixtures_df['competition'].unique())
            selected_competition = st.selectbox("Competition", competitions)
    
    with col3:
        if 'home_team' in fixtures_df.columns and 'away_team' in fixtures_df.columns:
            all_teams = list(set(list(fixtures_df['home_team'].unique()) + list(fixtures_df['away_team'].unique())))
            selected_team = st.selectbox("Team", ['All'] + sorted(all_teams))
    
    auto_refresh = st.checkbox("Auto-refresh (15 min interval)")
    
    st.markdown("---")
    
    st.header("📋 Match Schedule")
    
    filtered_df = fixtures_df.copy()
    
    if selected_competition != 'All' and 'competition' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['competition'] == selected_competition]
    
    if selected_team != 'All':
        if 'home_team' in filtered_df.columns and 'away_team' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['home_team'] == selected_team) | 
                (filtered_df['away_team'] == selected_team)
            ]
    
    if len(filtered_df) > 0:
        display_cols = []
        if 'date' in filtered_df.columns:
            display_cols.append('date')
        if 'home_team' in filtered_df.columns:
            display_cols.append('home_team')
        if 'away_team' in filtered_df.columns:
            display_cols.append('away_team')
        if 'venue' in filtered_df.columns:
            display_cols.append('venue')
        if 'status' in filtered_df.columns:
            display_cols.append('status')
        if 'prob_home_win' in filtered_df.columns:
            display_cols.extend(['prob_home_win', 'prob_draw', 'prob_away_win'])
        
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols],
            width='stretch',
            hide_index=True
        )
        
        st.markdown("---")
        
        st.header("🎯 Match Details")
        
        if len(filtered_df) > 0:
            match_options = [
                f"{row['home_team']} vs {row['away_team']}" 
                if 'home_team' in row and 'away_team' in row 
                else f"Match {idx}"
                for idx, row in filtered_df.iterrows()
            ]
            
            selected_match = st.selectbox("Select Match", match_options)
            
            if selected_match:
                match_idx = match_options.index(selected_match)
                match_data = filtered_df.iloc[match_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Match Information")
                    st.write(f"**Date:** {match_data.get('date', 'N/A')}")
                    st.write(f"**Venue:** {match_data.get('venue', 'N/A')}")
                    st.write(f"**City:** {match_data.get('city', 'N/A')}")
                    st.write(f"**Status:** {match_data.get('status', 'N/A')}")
                    
                    if 'temp_c' in match_data:
                        st.write(f"**Temperature:** {match_data['temp_c']}°C")
                    if 'humidity' in match_data:
                        st.write(f"**Humidity:** {match_data['humidity']}%")
                
                with col2:
                    st.subheader("Win Probabilities")
                    
                    if all(col in match_data for col in ['prob_home_win', 'prob_draw', 'prob_away_win']):
                        prob_data = pd.DataFrame({
                            'Outcome': [
                                f"{match_data.get('home_team', 'Home')} Win",
                                'Draw',
                                f"{match_data.get('away_team', 'Away')} Win"
                            ],
                            'Probability': [
                                match_data['prob_home_win'] * 100,
                                match_data['prob_draw'] * 100,
                                match_data['prob_away_win'] * 100
                            ]
                        })
                        
                        fig = px.bar(
                            prob_data,
                            x='Outcome',
                            y='Probability',
                            color='Probability',
                            color_continuous_scale='Blues',
                            text='Probability'
                        )
                        
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig.update_layout(
                            title="Match Outcome Probabilities",
                            yaxis_title="Probability (%)",
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Probability predictions not available for this match")
                
                st.markdown("---")
                
                st.subheader("📊 Head-to-Head History")
                st.info("Historical head-to-head data will be displayed here")
    
    else:
        st.warning("No fixtures found matching the selected filters")
    
    st.markdown("---")
    
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if auto_refresh:
        st.caption("Auto-refresh enabled (15 min interval)")
