"""
Outlier Detection for Outstanding NBA Players
Implements multiple techniques to identify exceptional players
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Directories
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


class PlayerOutlierDetector:
    """Detect outstanding players using multiple outlier detection methods"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.players_data = None
        self.features = None
        self.outlier_scores = {}
        
    def load_and_prepare_data(self):
        """Load and prepare player data for analysis"""
        print("Loading player data...")
        
        # Load datasets
        players = pd.read_csv(DATA_DIR / 'players.txt', encoding='latin-1')
        player_season = pd.read_csv(DATA_DIR / 'player_regular_season.txt', encoding='latin-1')
        
        # Calculate career statistics - aggregate by player
        career_stats = player_season.groupby('ilkid').agg({
            'gp': 'sum',           # Total games played
            'pts': 'sum',          # Total points
            'reb': 'sum',          # Total rebounds
            'asts': 'sum',         # Total assists
            'stl': 'sum',          # Total steals
            'blk': 'sum',          # Total blocks
            'fgm': 'sum',          # Field goals made
            'fga': 'sum',          # Field goals attempted
            'ftm': 'sum',          # Free throws made
            'fta': 'sum',          # Free throws attempted
            'minutes': 'sum',      # Total minutes
            'year': ['min', 'max'] # Career span
        })
        
        # Flatten multi-level columns
        new_cols = []
        for col in career_stats.columns:
            if isinstance(col, tuple):
                # For multi-level columns like ('year', 'min')
                new_cols.append('_'.join(str(c) for c in col if c))
            else:
                # For single-level columns
                new_cols.append(col)
        career_stats.columns = new_cols
        career_stats = career_stats.reset_index()
        
        # Rename columns to remove _sum suffix
        career_stats = career_stats.rename(columns={
            'gp_sum': 'gp', 'pts_sum': 'pts', 'reb_sum': 'reb',
            'asts_sum': 'asts', 'stl_sum': 'stl', 'blk_sum': 'blk',
            'fgm_sum': 'fgm', 'fga_sum': 'fga', 'ftm_sum': 'ftm',
            'fta_sum': 'fta', 'minutes_sum': 'minutes'
        })
        
        # Calculate advanced metrics
        career_stats['ppg'] = career_stats['pts'] / career_stats['gp']
        career_stats['rpg'] = career_stats['reb'] / career_stats['gp']
        career_stats['apg'] = career_stats['asts'] / career_stats['gp']
        career_stats['spg'] = career_stats['stl'] / career_stats['gp']
        career_stats['bpg'] = career_stats['blk'] / career_stats['gp']
        career_stats['mpg'] = career_stats['minutes'] / career_stats['gp']
        career_stats['fg_pct'] = (career_stats['fgm'] / career_stats['fga'] * 100).fillna(0)
        career_stats['ft_pct'] = (career_stats['ftm'] / career_stats['fta'] * 100).fillna(0)
        career_stats['career_length'] = career_stats['year_max'] - career_stats['year_min'] + 1
        
        # Merge with player info
        self.players_data = career_stats.merge(
            players[['ilkid', 'firstname', 'lastname', 'position']], 
            on='ilkid', 
            how='left'
        )
        self.players_data['fullname'] = (
            self.players_data['firstname'] + ' ' + self.players_data['lastname']
        )
        
        # Filter players with significant playing time
        self.players_data = self.players_data[
            (self.players_data['gp'] >= 100) & 
            (self.players_data['mpg'] >= 10)
        ].copy()
        
        print(f"✓ Loaded data for {len(self.players_data)} players")
        return self.players_data
    
    def detect_statistical_outliers(self, contamination=0.05):
        """Detect outliers using Z-score method"""
        print("\n1. Statistical Outliers (Z-score method)...")
        
        # Select features for analysis
        feature_cols = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'fg_pct', 'ft_pct', 
                       'pts', 'career_length', 'gp']
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(self.players_data[feature_cols].fillna(0)))
        
        # Identify outliers (Z-score > 3 in any feature)
        outlier_mask = (z_scores > 3).any(axis=1)
        self.outlier_scores['z_score'] = outlier_mask.astype(int)
        
        outliers = self.players_data[outlier_mask].copy()
        outliers['max_z_score'] = z_scores.max(axis=1)[outlier_mask]
        outliers = outliers.sort_values('max_z_score', ascending=False)
        
        print(f"   Found {len(outliers)} statistical outliers")
        return outliers
    
    def detect_isolation_forest(self, contamination=0.05):
        """Detect outliers using Isolation Forest"""
        print("\n2. Isolation Forest method...")
        
        feature_cols = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'fg_pct', 'ft_pct', 
                       'pts', 'career_length', 'gp']
        X = self.players_data[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)
        
        # Outliers are labeled as -1
        outlier_mask = predictions == -1
        self.outlier_scores['isolation_forest'] = outlier_mask.astype(int)
        
        outliers = self.players_data[outlier_mask].copy()
        outliers['anomaly_score'] = scores[outlier_mask]
        outliers = outliers.sort_values('anomaly_score')
        
        print(f"   Found {len(outliers)} outliers")
        return outliers
    
    def detect_local_outlier_factor(self, contamination=0.05):
        """Detect outliers using Local Outlier Factor"""
        print("\n3. Local Outlier Factor method...")
        
        feature_cols = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'fg_pct', 'ft_pct', 
                       'pts', 'career_length', 'gp']
        X = self.players_data[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=contamination, novelty=False)
        predictions = lof.fit_predict(X_scaled)
        scores = lof.negative_outlier_factor_
        
        # Outliers are labeled as -1
        outlier_mask = predictions == -1
        self.outlier_scores['lof'] = outlier_mask.astype(int)
        
        outliers = self.players_data[outlier_mask].copy()
        outliers['lof_score'] = scores[outlier_mask]
        outliers = outliers.sort_values('lof_score')
        
        print(f"   Found {len(outliers)} outliers")
        return outliers
    
    def detect_elliptic_envelope(self, contamination=0.05):
        """Detect outliers using Elliptic Envelope (assumes Gaussian distribution)"""
        print("\n4. Elliptic Envelope method...")
        
        feature_cols = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'fg_pct', 'ft_pct', 
                       'pts', 'career_length', 'gp']
        X = self.players_data[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Elliptic Envelope
        envelope = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = envelope.fit_predict(X_scaled)
        scores = envelope.score_samples(X_scaled)
        
        # Outliers are labeled as -1
        outlier_mask = predictions == -1
        self.outlier_scores['elliptic_envelope'] = outlier_mask.astype(int)
        
        outliers = self.players_data[outlier_mask].copy()
        outliers['mahalanobis_score'] = scores[outlier_mask]
        outliers = outliers.sort_values('mahalanobis_score')
        
        print(f"   Found {len(outliers)} outliers")
        return outliers
    
    def get_consensus_outliers(self, min_methods=2):
        """Get players identified as outliers by multiple methods"""
        print(f"\n5. Consensus outliers (detected by >= {min_methods} methods)...")
        
        # Create dataframe with all scores - reset index to match players_data
        scores_df = pd.DataFrame(self.outlier_scores).reset_index(drop=True)
        scores_df['total_methods'] = scores_df.sum(axis=1)
        
        # Add scores to players_data
        players_with_scores = self.players_data.reset_index(drop=True).copy()
        players_with_scores['methods_count'] = scores_df['total_methods']
        
        # Get consensus outliers
        consensus_outliers = players_with_scores[
            players_with_scores['methods_count'] >= min_methods
        ].copy()
        consensus_outliers = consensus_outliers.sort_values('methods_count', ascending=False)
        
        print(f"   Found {len(consensus_outliers)} consensus outliers")
        return consensus_outliers
    
    def print_top_performers(self, consensus_outliers, top_n=20):
        """Print top performers to console instead of creating visualizations"""
        print("\n" + "="*80)
        print("6. TOP PERFORMERS BY CATEGORY")
        print("="*80)
        
        # Top Scorers
        print("\n" + "-"*80)
        print(f"TOP {top_n} PLAYERS BY POINTS PER GAME:")
        print("-"*80)
        top_scorers = self.players_data.nlargest(top_n, 'ppg')
        for i, (_, player) in enumerate(top_scorers.iterrows(), 1):
            print(f"{i:2d}. {player['fullname']:30s} - {player['ppg']:5.1f} PPG  ({int(player['gp'])} games, {int(player['pts'])} pts)")
        
        # Top Rebounders
        print("\n" + "-"*80)
        print(f"TOP {top_n} PLAYERS BY REBOUNDS PER GAME:")
        print("-"*80)
        top_rebounders = self.players_data.nlargest(top_n, 'rpg')
        for i, (_, player) in enumerate(top_rebounders.iterrows(), 1):
            print(f"{i:2d}. {player['fullname']:30s} - {player['rpg']:5.1f} RPG  ({int(player['gp'])} games, {int(player['reb'])} rebs)")
        
        # Top Assist Leaders
        print("\n" + "-"*80)
        print(f"TOP {top_n} PLAYERS BY ASSISTS PER GAME:")
        print("-"*80)
        top_assisters = self.players_data.nlargest(top_n, 'apg')
        for i, (_, player) in enumerate(top_assisters.iterrows(), 1):
            print(f"{i:2d}. {player['fullname']:30s} - {player['apg']:5.1f} APG  ({int(player['gp'])} games, {int(player['asts'])} asts)")
        
        # Top Career Scorers
        print("\n" + "-"*80)
        print(f"TOP {top_n} CAREER SCORING LEADERS:")
        print("-"*80)
        top_overall = self.players_data.nlargest(top_n, 'pts')
        for i, (_, player) in enumerate(top_overall.iterrows(), 1):
            print(f"{i:2d}. {player['fullname']:30s} - {int(player['pts']):,} points  ({int(player['gp'])} games, {player['ppg']:.1f} PPG)")
        
        # Average comparison
        if len(consensus_outliers) > 0:
            print("\n" + "-"*80)
            print("AVERAGE PERFORMANCE COMPARISON:")
            print("-"*80)
            avg_all_ppg = self.players_data['ppg'].mean()
            avg_all_rpg = self.players_data['rpg'].mean()
            avg_all_apg = self.players_data['apg'].mean()
            avg_out_ppg = consensus_outliers['ppg'].mean()
            avg_out_rpg = consensus_outliers['rpg'].mean()
            avg_out_apg = consensus_outliers['apg'].mean()
            
            print(f"{'Metric':<20} {'Average Player':>15} {'Outstanding Player':>20} {'Difference':>15}")
            print(f"{'-'*20} {'-'*15} {'-'*20} {'-'*15}")
            print(f"{'Points Per Game':<20} {avg_all_ppg:>15.2f} {avg_out_ppg:>20.2f} {avg_out_ppg-avg_all_ppg:>14.2f}x")
            print(f"{'Rebounds Per Game':<20} {avg_all_rpg:>15.2f} {avg_out_rpg:>20.2f} {avg_out_rpg-avg_all_rpg:>14.2f}x")
            print(f"{'Assists Per Game':<20} {avg_all_apg:>15.2f} {avg_out_apg:>20.2f} {avg_out_apg-avg_all_apg:>14.2f}x")
    
    def generate_report(self, consensus_outliers):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("OBJECTIVE 1: OUTSTANDING PLAYERS DETECTION - OUTLIER ANALYSIS")
        print("="*80)
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Total players analyzed: {len(self.players_data):,}")
        print(f"   Outstanding players identified: {len(consensus_outliers)}")
        print(f"   Percentage: {len(consensus_outliers)/len(self.players_data)*100:.2f}%")
        
        # Method breakdown
        print("\n🔍 DETECTION METHOD BREAKDOWN:")
        for method, scores in self.outlier_scores.items():
            count = sum(scores)
            print(f"   {method}: {count} outliers detected")
        
        # Consensus distribution
        if 'methods_count' in consensus_outliers.columns:
            print("\n📈 CONSENSUS DISTRIBUTION:")
            for count in sorted(consensus_outliers['methods_count'].unique(), reverse=True):
                num_players = len(consensus_outliers[consensus_outliers['methods_count'] == count])
                print(f"   Detected by {int(count)}/4 methods: {num_players} players")
        
        print("\n" + "-"*80)
        print("TOP 20 OUTSTANDING PLAYERS (Ranked by consensus):")
        print("-"*80)
        print(f"{'Rank':<5} {'Player Name':<25} {'Pos':<4} {'Methods':<8} {'PPG':<6} {'RPG':<6} {'APG':<6} {'Career Pts':<12}")
        print("-"*80)
        
        top_20 = consensus_outliers.head(20)
        for rank, (idx, row) in enumerate(top_20.iterrows(), 1):
            print(f"{rank:<5} {row['fullname'][:24]:<25} {row['position']:<4} "
                  f"{int(row['methods_count'])}/4{'':<4} {row['ppg']:<6.1f} {row['rpg']:<6.1f} "
                  f"{row['apg']:<6.1f} {int(row['pts']):>11,}")
        
        # Category leaders
        print("\n" + "-"*80)
        print("CATEGORY LEADERS AMONG OUTSTANDING PLAYERS:")
        print("-"*80)
        
        print("\n🏀 SCORING LEADERS (PPG):")
        top_scorers = consensus_outliers.nlargest(5, 'ppg')
        for idx, row in top_scorers.iterrows():
            print(f"   {row['fullname']:<30} {row['ppg']:>5.1f} PPG")
        
        print("\n🏀 REBOUNDING LEADERS (RPG):")
        top_rebounders = consensus_outliers.nlargest(5, 'rpg')
        for idx, row in top_rebounders.iterrows():
            print(f"   {row['fullname']:<30} {row['rpg']:>5.1f} RPG")
        
        print("\n🏀 PLAYMAKING LEADERS (APG):")
        top_assisters = consensus_outliers.nlargest(5, 'apg')
        for idx, row in top_assisters.iterrows():
            print(f"   {row['fullname']:<30} {row['apg']:>5.1f} APG")
        
        print("\n🏀 CAREER SCORING LEADERS:")
        top_career = consensus_outliers.nlargest(5, 'pts')
        for idx, row in top_career.iterrows():
            print(f"   {row['fullname']:<30} {int(row['pts']):>11,} points")
        
        # Statistical comparison
        print("\n" + "-"*80)
        print("STATISTICAL COMPARISON: Outstanding vs Average Players")
        print("-"*80)
        avg_all = self.players_data[['ppg', 'rpg', 'apg', 'fg_pct', 'career_length', 'gp']].mean()
        avg_outstanding = consensus_outliers[['ppg', 'rpg', 'apg', 'fg_pct', 'career_length', 'gp']].mean()
        
        metrics = ['ppg', 'rpg', 'apg', 'fg_pct', 'career_length', 'gp']
        labels = ['Points Per Game', 'Rebounds Per Game', 'Assists Per Game', 
                 'Field Goal %', 'Career Length (years)', 'Games Played']
        
        for metric, label in zip(metrics, labels):
            avg_val = avg_all[metric]
            out_val = avg_outstanding[metric]
            diff = ((out_val - avg_val) / avg_val * 100) if avg_val > 0 else 0
            print(f"   {label:<25} Average: {avg_val:>7.1f}  Outstanding: {out_val:>7.1f}  "
                  f"({diff:+.1f}%)")
        
        print("\n" + "="*80)
        
        # Save detailed results
        results_file = OUTPUT_DIR / 'outstanding_players_results.csv'
        consensus_outliers.to_csv(results_file, index=False)
        print(f"\n✓ Detailed results saved to: {results_file}")


def main():
    """Main function to run outlier detection"""
    print("="*70)
    print("NBA OUTSTANDING PLAYERS - OUTLIER DETECTION")
    print("="*70)
    
    # Initialize detector
    detector = PlayerOutlierDetector()
    
    # Load data
    detector.load_and_prepare_data()
    
    # Run multiple outlier detection methods
    z_score_outliers = detector.detect_statistical_outliers(contamination=0.05)
    iso_forest_outliers = detector.detect_isolation_forest(contamination=0.05)
    lof_outliers = detector.detect_local_outlier_factor(contamination=0.05)
    elliptic_outliers = detector.detect_elliptic_envelope(contamination=0.05)
    
    # Get consensus outliers
    consensus_outliers = detector.get_consensus_outliers(min_methods=2)
    
    # Print top performers
    detector.print_top_performers(consensus_outliers, top_n=20)
    
    # Generate report
    detector.generate_report(consensus_outliers)
    
    print("\n" + "="*70)
    print("✅ OUTLIER DETECTION COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  1. output/outstanding_players_results.csv")
    print()


if __name__ == "__main__":
    main()
