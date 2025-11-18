"""
Game Outcome Prediction
Predict basketball game outcomes given two teams using ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Directories
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


class GameOutcomePredictor:
    """Predict game outcomes using multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.team_stats = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self):
        """Load and prepare team data for game outcome prediction"""
        print("Loading team data...")
        
        # Load team season data
        team_season = pd.read_csv(DATA_DIR / 'team_season.txt', encoding='latin-1')
        
        # Calculate team performance metrics
        team_season['total_games'] = team_season['won'] + team_season['lost']
        
        # Filter out teams with 0 games (invalid data)
        team_season = team_season[team_season['total_games'] > 0].copy()
        
        team_season['win_pct'] = team_season['won'] / team_season['total_games']
        
        # Offensive efficiency (points per game)
        team_season['off_ppg'] = team_season['o_pts'] / team_season['total_games']
        team_season['def_ppg'] = team_season['d_pts'] / team_season['total_games']
        
        # Advanced metrics with safe division
        team_season['o_fg_pct'] = np.where(
            team_season['o_fga'] > 0, 
            team_season['o_fgm'] / team_season['o_fga'] * 100, 
            0
        )
        team_season['d_fg_pct'] = np.where(
            team_season['d_fga'] > 0,
            team_season['d_fgm'] / team_season['d_fga'] * 100,
            0
        )
        team_season['o_ft_pct'] = np.where(
            team_season['o_fta'] > 0,
            team_season['o_ftm'] / team_season['o_fta'] * 100,
            0
        )
        team_season['o_3p_pct'] = np.where(
            team_season['o_3pa'] > 0,
            team_season['o_3pm'] / team_season['o_3pa'] * 100,
            0
        )
        
        # Rebound rates
        team_season['o_reb_pg'] = team_season['o_reb'] / team_season['total_games']
        team_season['d_reb_pg'] = team_season['d_reb'] / team_season['total_games']
        
        # Assist and turnover rates
        team_season['o_ast_pg'] = team_season['o_asts'] / team_season['total_games']
        team_season['o_to_pg'] = team_season['o_to'] / team_season['total_games']
        team_season['d_to_pg'] = team_season['d_to'] / team_season['total_games']
        
        # Net ratings
        team_season['net_rating'] = team_season['off_ppg'] - team_season['def_ppg']
        
        # Replace any remaining inf/nan values
        team_season = team_season.replace([np.inf, -np.inf], 0).fillna(0)
        
        self.team_stats = team_season
        print(f"✓ Loaded {len(team_season)} team-seasons")
        
        return team_season
    
    def create_matchup_dataset(self):
        """Create synthetic matchup dataset from team statistics"""
        print("\nCreating matchup dataset...")
        
        # Get teams from recent years for more relevant data
        recent_teams = self.team_stats[self.team_stats['year'] >= 1990].copy()
        
        matchups = []
        
        # Create all possible matchups within same year
        for year in recent_teams['year'].unique():
            year_teams = recent_teams[recent_teams['year'] == year]
            
            # Create matchups
            for idx1, team1 in year_teams.iterrows():
                for idx2, team2 in year_teams.iterrows():
                    if team1['team'] != team2['team']:
                        # Team 1 features (home team)
                        matchup = {
                            'year': year,
                            'home_team': team1['team'],
                            'away_team': team2['team'],
                            'home_win_pct': team1['win_pct'],
                            'away_win_pct': team2['win_pct'],
                            'home_off_ppg': team1['off_ppg'],
                            'away_off_ppg': team2['off_ppg'],
                            'home_def_ppg': team1['def_ppg'],
                            'away_def_ppg': team2['def_ppg'],
                            'home_o_fg_pct': team1['o_fg_pct'],
                            'away_o_fg_pct': team2['o_fg_pct'],
                            'home_o_3p_pct': team1['o_3p_pct'],
                            'away_o_3p_pct': team2['o_3p_pct'],
                            'home_o_reb_pg': team1['o_reb_pg'],
                            'away_o_reb_pg': team2['o_reb_pg'],
                            'home_o_ast_pg': team1['o_ast_pg'],
                            'away_o_ast_pg': team2['o_ast_pg'],
                            'home_o_to_pg': team1['o_to_pg'],
                            'away_o_to_pg': team2['o_to_pg'],
                            'home_net_rating': team1['net_rating'],
                            'away_net_rating': team2['net_rating'],
                        }
                        
                        # Determine winner based on win percentage and home advantage
                        # Add home advantage factor (typically ~5% boost)
                        home_strength = team1['win_pct'] * 1.05 + team1['net_rating'] * 0.01
                        away_strength = team2['win_pct'] + team2['net_rating'] * 0.01
                        
                        matchup['home_win'] = 1 if home_strength > away_strength else 0
                        
                        matchups.append(matchup)
        
        matchup_df = pd.DataFrame(matchups)
        print(f"✓ Created {len(matchup_df)} matchups")
        print(f"   Home team wins: {matchup_df['home_win'].sum()} ({matchup_df['home_win'].mean()*100:.1f}%)")
        
        return matchup_df
    
    def prepare_features(self, matchup_df):
        """Prepare features for ML models"""
        print("\nPreparing features...")
        
        # Feature columns
        feature_cols = [
            'home_win_pct', 'away_win_pct',
            'home_off_ppg', 'away_off_ppg',
            'home_def_ppg', 'away_def_ppg',
            'home_o_fg_pct', 'away_o_fg_pct',
            'home_o_3p_pct', 'away_o_3p_pct',
            'home_o_reb_pg', 'away_o_reb_pg',
            'home_o_ast_pg', 'away_o_ast_pg',
            'home_o_to_pg', 'away_o_to_pg',
            'home_net_rating', 'away_net_rating'
        ]
        
        X = matchup_df[feature_cols]
        y = matchup_df['home_win']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n1. Training Logistic Regression...")
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return model, accuracy
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n2. Training Random Forest...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return model, accuracy
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n3. Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        self.models['Gradient Boosting'] = model
        self.results['Gradient Boosting'] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return model, accuracy
    
    def train_svm(self):
        """Train Support Vector Machine model"""
        print("\n4. Training SVM...")
        
        model = SVC(kernel='rbf', random_state=42)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        self.models['SVM'] = model
        self.results['SVM'] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return model, accuracy
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n5. Training XGBoost...")
        
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return model, accuracy
    
    def train_lightgbm(self):
        """Train LightGBM model"""
        print("\n6. Training LightGBM...")
        
        model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        self.models['LightGBM'] = model
        self.results['LightGBM'] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return model, accuracy
    
    def print_results(self):
        """Print model comparison results to console"""
        print("\n" + "="*80)
        print("7. MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Model ranking by test accuracy
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("\n📊 MODEL RANKINGS BY TEST ACCURACY:")
        print(f"{'Rank':<6} {'Model':<20} {'Test Accuracy':<15} {'CV Mean':<12} {'CV Std':<10}")
        print("-" * 80)
        
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            print(f"{rank:<6} {model_name:<20} {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)  "
                  f"{results['cv_mean']:.4f}      {results['cv_std']:.4f}")
        
        # Best model details
        best_model_name = sorted_models[0][0]
        best_results = sorted_models[0][1]
        best_predictions = best_results['predictions']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Test Accuracy: {best_results['accuracy']:.4f} ({best_results['accuracy']*100:.2f}%)")
        print(f"   CV Mean: {best_results['cv_mean']:.4f} ± {best_results['cv_std']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, best_predictions)
        print(f"\n📋 CONFUSION MATRIX ({best_model_name}):")
        print(f"{'':>15} {'Predicted Away Win':<20} {'Predicted Home Win':<20}")
        print(f"{'Actual Away Win':<15} {cm[0, 0]:<20} {cm[0, 1]:<20}")
        print(f"{'Actual Home Win':<15} {cm[1, 0]:<20} {cm[1, 1]:<20}")
        
        # Calculate derived metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        away_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        away_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        away_f1 = 2 * (away_precision * away_recall) / (away_precision + away_recall) if (away_precision + away_recall) > 0 else 0
        
        home_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        home_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        home_f1 = 2 * (home_precision * home_recall) / (home_precision + home_recall) if (home_precision + home_recall) > 0 else 0
        
        # Classification Report
        print(f"\n📈 CLASSIFICATION METRICS ({best_model_name}):")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        print(f"{'Away Win':<15} {away_precision:.4f}       {away_recall:.4f}       {away_f1:.4f}")
        print(f"{'Home Win':<15} {home_precision:.4f}       {home_recall:.4f}       {home_f1:.4f}")
        
        # Feature Importance (Random Forest)
        if 'Random Forest' in self.models:
            print(f"\n🎯 TOP 10 FEATURE IMPORTANCES (Random Forest):")
            feature_names = [
                'Home Win%', 'Away Win%', 'Home Off PPG', 'Away Off PPG',
                'Home Def PPG', 'Away Def PPG', 'Home FG%', 'Away FG%',
                'Home 3P%', 'Away 3P%', 'Home Reb/G', 'Away Reb/G',
                'Home Ast/G', 'Away Ast/G', 'Home TO/G', 'Away TO/G',
                'Home Net', 'Away Net'
            ]
            importances = self.models['Random Forest'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            print(f"{'Rank':<6} {'Feature':<20} {'Importance':<15}")
            print("-" * 80)
            for rank, idx in enumerate(indices, 1):
                print(f"{rank:<6} {feature_names[idx]:<20} {importances[idx]:.6f}")
        
        # Cross-Validation Performance Summary
        print(f"\n🔄 CROSS-VALIDATION SUMMARY (5-Fold):")
        print(f"{'Model':<20} {'Mean Accuracy':<15} {'Std Dev':<12} {'Range':<20}")
        print("-" * 80)
        
        for model_name, results in sorted_models:
            cv_min = results['cv_mean'] - results['cv_std']
            cv_max = results['cv_mean'] + results['cv_std']
            print(f"{model_name:<20} {results['cv_mean']:.4f}          {results['cv_std']:.4f}      "
                  f"{cv_min:.4f} - {cv_max:.4f}")
    
    def generate_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("OBJECTIVE 2: GAME OUTCOME PREDICTION - MACHINE LEARNING MODELS")
        print("="*80)
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Total matchups analyzed: {len(self.X_train) + len(self.X_test):,}")
        print(f"   Training samples: {len(self.X_train):,}")
        print(f"   Test samples: {len(self.X_test):,}")
        print(f"   Features used: {self.X_train.shape[1]}")
        home_win_rate = (self.y_train.sum() + self.y_test.sum()) / (len(self.y_train) + len(self.y_test))
        print(f"   Home team win rate: {home_win_rate*100:.1f}%")
        
        print("\n" + "-"*80)
        print("MODEL PERFORMANCE COMPARISON (6 ML Algorithms):")
        print("-"*80)
        print(f"{'Rank':<5} {'Model':<25} {'Test Acc':<12} {'CV Acc':<15} {'CV Std':<10}")
        print("-"*80)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            print(f"{rank:<5} {model_name:<25} {metrics['accuracy']:<12.4f} "
                  f"{metrics['cv_mean']:<15.4f} {metrics['cv_std']:<10.4f}")
        
        # Best model details
        best_model_name = sorted_results[0][0]
        best_predictions = sorted_results[0][1]['predictions']
        best_accuracy = sorted_results[0][1]['accuracy']
        
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print("="*80)
        print(f"   Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"   CV Accuracy: {sorted_results[0][1]['cv_mean']:.4f}")
        
        print("\n📈 DETAILED CLASSIFICATION REPORT:")
        print("-"*80)
        print(classification_report(self.y_test, best_predictions, 
                                   target_names=['Away Win', 'Home Win']))
        
        # Confusion Matrix Details
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, best_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n📊 CONFUSION MATRIX BREAKDOWN:")
        print("-"*80)
        print(f"   True Negatives (Correct Away Win predictions): {tn:,}")
        print(f"   True Positives (Correct Home Win predictions):  {tp:,}")
        print(f"   False Negatives (Missed Home Wins):             {fn:,}")
        print(f"   False Positives (Incorrect Home Win calls):     {fp:,}")
        print(f"\n   Total Correct Predictions: {tn + tp:,} out of {len(self.y_test):,} "
              f"({(tn + tp)/len(self.y_test)*100:.2f}%)")
        
        # Feature importance if available
        if 'Random Forest' in self.models:
            print("\n" + "-"*80)
            print("🎯 TOP 10 MOST IMPORTANT FEATURES (Random Forest):")
            print("-"*80)
            
            feature_names = [
                'Home Win%', 'Away Win%', 'Home Off PPG', 'Away Off PPG',
                'Home Def PPG', 'Away Def PPG', 'Home FG%', 'Away FG%',
                'Home 3P%', 'Away 3P%', 'Home Reb/G', 'Away Reb/G',
                'Home Ast/G', 'Away Ast/G', 'Home TO/G', 'Away TO/G',
                'Home Net', 'Away Net'
            ]
            importances = self.models['Random Forest'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices, 1):
                print(f"   {i:2d}. {feature_names[idx]:<20} {importances[idx]:.4f} "
                      f"({'▓' * int(importances[idx] * 50)})")
        
        # Model comparison insights
        print("\n" + "-"*80)
        print("💡 KEY INSIGHTS:")
        print("-"*80)
        
        # Find best and worst models
        accuracies = [m[1]['accuracy'] for m in sorted_results]
        accuracy_range = max(accuracies) - min(accuracies)
        
        print(f"   • Best performing model: {sorted_results[0][0]} ({sorted_results[0][1]['accuracy']:.4f})")
        print(f"   • Lowest performing model: {sorted_results[-1][0]} ({sorted_results[-1][1]['accuracy']:.4f})")
        print(f"   • Accuracy range across models: {accuracy_range:.4f} ({accuracy_range*100:.2f}%)")
        print(f"   • Average accuracy across all models: {np.mean(accuracies):.4f}")
        
        # Performance categories
        ensemble_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
        ensemble_acc = [self.results[m]['accuracy'] for m in ensemble_models if m in self.results]
        if ensemble_acc:
            print(f"   • Ensemble models average: {np.mean(ensemble_acc):.4f}")
        
        linear_models = ['Logistic Regression', 'SVM']
        linear_acc = [self.results[m]['accuracy'] for m in linear_models if m in self.results]
        if linear_acc:
            print(f"   • Linear models average: {np.mean(linear_acc):.4f}")
        
        # Prediction confidence
        if hasattr(self.models[best_model_name], 'predict_proba'):
            probas = self.models[best_model_name].predict_proba(self.X_test)
            avg_confidence = np.mean(np.max(probas, axis=1))
            print(f"   • Average prediction confidence: {avg_confidence:.4f} ({avg_confidence*100:.1f}%)")
        
        print("\n" + "="*80)
        
        # Save results
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Test_Accuracy': metrics['accuracy'],
                'CV_Mean': metrics['cv_mean'],
                'CV_Std': metrics['cv_std']
            }
            for name, metrics in self.results.items()
        ]).sort_values('Test_Accuracy', ascending=False)
        
        results_file = OUTPUT_DIR / 'prediction_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to: {results_file}")
    
    def predict_game(self, home_team_stats, away_team_stats, model_name='XGBoost'):
        """Predict outcome of a specific game"""
        if model_name not in self.models:
            model_name = list(self.models.keys())[0]
        
        model = self.models[model_name]
        
        # Prepare features
        features = np.array([[
            home_team_stats['win_pct'], away_team_stats['win_pct'],
            home_team_stats['off_ppg'], away_team_stats['off_ppg'],
            home_team_stats['def_ppg'], away_team_stats['def_ppg'],
            home_team_stats['o_fg_pct'], away_team_stats['o_fg_pct'],
            home_team_stats['o_3p_pct'], away_team_stats['o_3p_pct'],
            home_team_stats['o_reb_pg'], away_team_stats['o_reb_pg'],
            home_team_stats['o_ast_pg'], away_team_stats['o_ast_pg'],
            home_team_stats['o_to_pg'], away_team_stats['o_to_pg'],
            home_team_stats['net_rating'], away_team_stats['net_rating']
        ]])
        
        features_scaled = self.scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            return prediction, probabilities
        
        return prediction, None


def main():
    """Main function to run game outcome prediction"""
    print("="*70)
    print("NBA GAME OUTCOME PREDICTION")
    print("="*70)
    
    # Initialize predictor
    predictor = GameOutcomePredictor()
    
    # Load and prepare data
    predictor.load_and_prepare_data()
    matchup_df = predictor.create_matchup_dataset()
    predictor.prepare_features(matchup_df)
    
    # Train multiple models
    print("\nTraining models...")
    print("="*70)
    predictor.train_logistic_regression()
    predictor.train_random_forest()
    predictor.train_gradient_boosting()
    predictor.train_svm()
    predictor.train_xgboost()
    predictor.train_lightgbm()
    
    # Print results
    predictor.print_results()
    
    # Generate report
    predictor.generate_report()
    
    print("\n" + "="*70)
    print("✅ GAME OUTCOME PREDICTION COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  1. output/prediction_results.csv")
    print()


if __name__ == "__main__":
    main()
