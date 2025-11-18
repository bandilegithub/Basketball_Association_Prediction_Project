# NBA Basketball Association - Machine Learning Analysis

## Project Overview

This project applies multiple Machine Learning techniques to analyze NBA/ABA basketball statistics from 1946-2005. The analysis focuses on two main objectives:

1. **Outstanding Player Detection**: Using outlier detection methods to identify exceptional players
2. **Game Outcome Prediction**: Predicting game results given two teams using various ML models

## Dataset

The dataset (`databasebasketball.zip`) contains comprehensive NBA and ABA statistics including:
- Player regular season statistics
- Player playoff statistics
- Player All-Star game statistics
- Team regular season statistics
- Complete draft history
- Coaching records

**Data Coverage**: 1946-2005
**Total Players**: 3,572
**Total Games**: 19,112 player-seasons

## Project Structure

```
Basketball_Association_prediction/
├── data/                          # Dataset files
│   ├── players.txt
│   ├── player_regular_season.txt
│   ├── player_playoffs.txt
│   ├── team_season.txt
│   ├── draft.txt
│   └── ...
├── src/                           # Source code
│   ├── main.py                    # Main runner script
│   ├── outlier_detection.py      # Outstanding player detection
│   ├── game_prediction.py        # Game outcome prediction
│   └── data_visualization.py     # Data visualizations (optional)
├── output/                        # Generated results
│   └── *.csv                      # Results data (CSV files)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Basketball_Association_prediction
```

2. **Create virtual environment**
```bash
python -m venv .venv
```

3. **Activate virtual environment**
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Analysis
```bash
python src/main.py
```

### Run Individual Components

**Data Visualization Only**
```bash
python src/data_visualization.py
```

**Outlier Detection Only**
```bash
python src/outlier_detection.py
```

**Game Prediction Only**
```bash
python src/game_prediction.py
```

## Methods and Techniques

### 1. Outstanding Player Detection (Outlier Analysis)

Multiple outlier detection methods are used to identify exceptional players:

#### **Statistical Method (Z-Score)**
- Calculates Z-scores for key performance metrics
- Identifies players with Z-score > 3 in any category
- Simple and interpretable

#### **Isolation Forest**
- Ensemble-based anomaly detection
- Isolates outliers by randomly selecting features
- Effective for high-dimensional data

#### **Local Outlier Factor (LOF)**
- Density-based method
- Compares local density of points
- Identifies players unusual compared to neighbors

#### **Elliptic Envelope**
- Assumes Gaussian distribution
- Uses Mahalanobis distance
- Fits elliptic boundary around normal data

#### **Consensus Approach**
- Combines results from all methods
- Players identified by ≥2 methods are "consensus outliers"
- Reduces false positives

**Key Features Analyzed:**
- Points Per Game (PPG)
- Rebounds Per Game (RPG)
- Assists Per Game (APG)
- Steals/Blocks Per Game
- Field Goal/Free Throw Percentage
- Career Length and Total Points

### 2. Game Outcome Prediction

Six machine learning models predict game winners:

#### **Logistic Regression**
- Baseline linear model
- Fast training, interpretable
- Good for linearly separable data

#### **Random Forest**
- Ensemble of decision trees
- Handles non-linear relationships
- Provides feature importance

#### **Gradient Boosting**
- Sequential ensemble method
- Corrects previous model errors
- High accuracy potential

#### **Support Vector Machine (SVM)**
- Finds optimal hyperplane
- Effective in high dimensions
- Uses RBF kernel

#### **XGBoost**
- Advanced gradient boosting
- Regularization prevents overfitting
- Industry-standard performance

#### **LightGBM**
- Efficient gradient boosting
- Fast training on large datasets
- Leaf-wise tree growth

**Input Features:**
- Team win percentage
- Offensive/Defensive points per game
- Field goal percentage
- 3-point percentage
- Rebounds, assists, turnovers per game
- Net rating
- Home court advantage

**Evaluation Metrics:**
- Test accuracy
- 5-fold cross-validation scores
- Precision, Recall, F1-Score
- Confusion matrices

## Results

### Outstanding Players Detected

The consensus outlier detection identified exceptional players including:
- **High Scorers**: Players with exceptional PPG (30+)
- **Dominant Rebounders**: Players with RPG exceeding normal by 3+ standard deviations
- **Elite Playmakers**: Assist leaders with APG > 10
- **All-Around Stars**: Exceptional in multiple categories

### Game Prediction Performance

Model performance (typical results):
- **XGBoost**: ~85-90% accuracy
- **LightGBM**: ~85-88% accuracy
- **Random Forest**: ~83-87% accuracy
- **Gradient Boosting**: ~82-86% accuracy
- **SVM**: ~80-84% accuracy
- **Logistic Regression**: ~78-82% accuracy

**Most Important Features:**
1. Team win percentage
2. Net rating (point differential)
3. Offensive efficiency
4. Defensive efficiency
5. Home court advantage

## Output

### Console Output

All analysis results are displayed directly in the console with comprehensive formatting:

**Outlier Detection:**
- Top 20 players by PPG, RPG, APG
- Top 20 career scoring leaders
- Consensus outliers ranked by detection methods
- Category leaders among outstanding players
- Statistical comparisons (outstanding vs average players)

**Game Prediction:**
- Model rankings by test accuracy
- Best model details and confusion matrix
- Classification metrics (precision, recall, F1-score)
- Top 10 feature importances
- Cross-validation summary
- Model performance comparison table

### Data Files

CSV files saved to the `output/` folder:
- `outstanding_players_results.csv` - Detected outliers with scores and statistics
- `prediction_results.csv` - Model performance summary and predictions

## Technical Details

### Performance Considerations
- Data preprocessing includes scaling and normalization
- Cross-validation prevents overfitting
- Stratified sampling maintains class balance
- Feature engineering creates advanced metrics

### Model Selection
- Multiple models tested for robustness
- Ensemble methods generally outperform linear models
- XGBoost/LightGBM recommended for production

### Limitations
- Historical data may not reflect modern game
- Synthetic matchups used for prediction training
- Home advantage estimated at 5%
- Limited features (no player injuries, momentum, etc.)

## Future Enhancements

1. **Real-time predictions** using current season data
2. **Player performance forecasting** for upcoming seasons
3. **Team composition optimization** using player statistics
4. **Advanced metrics** (PER, True Shooting%, etc.)
5. **Deep learning models** for sequence prediction
6. **Web interface** for interactive predictions

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
xgboost>=2.0.0
lightgbm>=4.0.0
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is for educational purposes. Dataset sourced from basketballreference.com.

## Authors

Created as part of a Machine Learning course project.

## Acknowledgments

- Basketball Reference for the comprehensive dataset
- scikit-learn, XGBoost, and LightGBM communities
- NBA/ABA for the amazing game of basketball

## Contact

For questions or collaboration, please open an issue on GitHub.

---

**Note**: This project demonstrates ML techniques for educational purposes. Predictions are based on historical data and should not be used for gambling or commercial purposes.
