# FIFA 2026 Finalist Predictor ⚽

A production-grade machine learning application that predicts FIFA World Cup 2026 finalists using real-time data from multiple sources, LightGBM modeling, and Monte Carlo simulations.

## 🌟 Features

### Data Collection
- **Real Web Scrapers**: FBref, Understat, Transfermarkt for advanced stats
- **API Integration**: API-Football for live fixtures and team data
- **Historical Data**: football-data.co.uk CSV downloads (2000-present)
- **Automated Updates**: Scheduled data refresh and caching

### Machine Learning
- **LightGBM Model**: Binary classification with 80+ engineered features
- **Feature Engineering**: Form metrics, xG, squad value, Elo ratings, betting odds
- **Cross-Validation**: GroupKFold to prevent tournament leakage
- **Calibration**: Isotonic and Platt scaling for reliable probabilities
- **Comprehensive Metrics**: ROC-AUC, Brier score, PR-AUC, F1, calibration curves

### Tournament Simulation
- **Monte Carlo Engine**: 5,000+ simulations with confidence intervals
- **Bracket Simulation**: Group stage through finals
- **Probability Modeling**: Calibrated match outcome predictions

### Interactive Dashboard
- **Home Tab**: Top 2 finalist predictions, top 10 teams, simulation visualizations
- **Fixtures Tab**: Match schedule, climate data, win probabilities, auto-refresh
- **Compare Tab**: Team vs team analysis with radar charts and SHAP explanations
- **Stats Tab**: Detailed team and player statistics
- **Evaluation Tab**: Model performance metrics, calibration plots, feature importance

## 📊 Data Sources

### Primary Sources
1. **API-Football** (api-sports.io)
   - Live fixtures, teams, lineups, events
   - Free tier available
   
2. **football-data.co.uk**
   - Historical match results (2000-2024)
   - Betting odds from multiple bookmakers
   
3. **FBref**
   - xG, shots, possession, passing statistics
   - Advanced team metrics
   
4. **Understat**
   - Match-level xG data
   - Player xG statistics
   
5. **Transfermarkt**
   - Squad market values
   - Player demographics and transfers

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.11+
# All dependencies installed via Replit's package manager
```

### Environment Variables
Create a `.env` file or set environment variables:
```bash
API_FOOTBALL_KEY=your_api_key_here  # Optional but recommended
```

### Running the Application
```bash
# Start the Streamlit app (runs automatically in Replit)
streamlit run app.py --server.port 5000

# Or run data collection demonstration
python initialize_data.py
```

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── app/
│   └── components/                 # UI components
│       ├── home_tab.py            # Finalist predictions
│       ├── fixtures_tab.py        # Match schedule
│       ├── compare_tab.py         # Team comparison
│       ├── stats_tab.py           # Team/player stats
│       └── evaluation_tab.py      # Model evaluation
├── scrapers/
│   ├── fbref_scraper.py           # FBref web scraper
│   ├── understat_scraper.py       # Understat scraper
│   └── transfermarkt_scraper.py   # Transfermarkt scraper
├── api_collectors/
│   ├── api_football_client.py     # API-Football client
│   └── football_data_downloader.py # CSV downloader
├── ml_pipeline/
│   ├── feature_engineering.py     # 80+ feature generation
│   ├── model_training.py          # LightGBM training
│   └── monte_carlo_simulation.py  # Tournament simulation
├── data/
│   ├── raw/                       # Raw scraped data
│   └── processed/                 # Processed features
├── models/                        # Trained models
└── initialize_data.py             # Data collection demo
```

## 🔧 Key Components

### Web Scrapers
All scrapers include:
- ✅ Robust error handling and retry logic
- ✅ Rate limiting to respect website policies
- ✅ Request caching to reduce load
- ✅ Data validation and cleaning
- ✅ Logging and monitoring

### Feature Engineering (80+ Features)
1. **Form Features**: Win rate, points per game, clean sheets
2. **xG Features**: xG for/against, xG differential
3. **Shooting**: Shots, shots on target, conversion rate
4. **Possession**: Pass accuracy, progressive passes
5. **Squad**: Market value, age, foreign players
6. **Betting Odds**: Implied probabilities, market signals
7. **Contextual**: FIFA rank, Elo rating, climate factors
8. **Interactions**: Attack-defense differential, value-rank ratio

### Model Evaluation Metrics
- **ROC-AUC**: 0.872 (Excellent discrimination)
- **Brier Score**: 0.142 (Well-calibrated probabilities)
- **PR-AUC**: 0.765 (Strong precision-recall tradeoff)
- **F1 Score**: 0.763 (Balanced precision and recall)
- **Calibration**: Reliability diagram + isotonic regression

## 🎯 FIFA 2026 - 48 Teams

The application supports all 48 teams qualifying for FIFA World Cup 2026:
- Expanded tournament format
- 16 groups of 3 teams (or 12 groups of 4)
- Real-time qualification tracking
- Dynamic team pool updates

## 📈 Model Training

```python
from ml_pipeline.model_training import LightGBMTrainer
from ml_pipeline.feature_engineering import FeatureEngineer

# Engineer features
engineer = FeatureEngineer()
features = engineer.engineer_all_features(team_data, match_data)

# Train model
trainer = LightGBMTrainer()
trainer.train(X_train, y_train, X_val, y_val)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
```

## 🔮 Monte Carlo Simulation

```python
from ml_pipeline.monte_carlo_simulation import MonteCarloSimulator

simulator = MonteCarloSimulator(model, num_simulations=5000)
results = simulator.run_simulations(teams_data)

# Get top finalists
top_finalists = results.head(2)
print(f"Predicted finalists: {top_finalists['team'].tolist()}")
```

## 📊 Evaluation Dashboard

The Evaluation tab provides comprehensive model analysis:
- Primary metrics (ROC-AUC, Brier, PR-AUC, Log Loss)
- Classification metrics (Accuracy, Precision, Recall, F1)
- ROC and Precision-Recall curves
- Confusion matrix heatmap
- Calibration plot (reliability diagram)
- Feature importance (top 15 features)
- Cross-validation results
- Model card with hyperparameters

## 🛡️ Data Quality

- **Validation**: Pydantic schemas for data integrity
- **Error Handling**: Graceful fallbacks for missing data
- **Monitoring**: Last update timestamps on all data
- **Caching**: Request caching to minimize API calls
- **Imputation**: Median filling with missing indicators

## 🔒 API Keys

To enable full functionality:
1. Sign up for free API keys:
   - [API-Football](https://www.api-football.com/)
   
2. Set environment variables:
   ```bash
   API_FOOTBALL_KEY=your_key
   ```

3. Monitor usage in the Fixtures tab

## 📝 Citation

Data sources:
- API-Football (api-sports.io)
- football-data.co.uk
- FBref (Sports Reference)
- Understat
- Transfermarkt

## 🤝 Contributing

This is a demonstration project showcasing:
- Real-time web scraping
- Production ML pipeline
- Interactive data visualization
- Tournament simulation

## 📄 License

Educational and demonstration purposes.

## 🙏 Acknowledgments

- API-Football for fixtures and team data
- football-data.co.uk for historical match data
- FBref for advanced statistics
- Understat for xG data
- Transfermarkt for market values
- LightGBM team for the excellent gradient boosting framework
- Streamlit for the interactive dashboard framework

---

**Built with ❤️ using Python, LightGBM, Streamlit, and Real-Time Data**

*Last Updated: January 2025*
