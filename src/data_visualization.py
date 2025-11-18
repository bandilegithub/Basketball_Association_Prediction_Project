"""
Basketball Association Data Visualization
Comprehensive visualization of basketball statistics datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Directories
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'output'

# Clean and recreate output directory
if OUTPUT_DIR.exists():
    for file in OUTPUT_DIR.glob('*.png'):
        file.unlink()
    print("✓ Cleaned previous output files")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load all datasets"""
    print("Loading datasets...")
    
    data = {}
    data['players'] = pd.read_csv(DATA_DIR / 'players.txt', encoding='latin-1')
    data['player_regular_season'] = pd.read_csv(DATA_DIR / 'player_regular_season.txt', encoding='latin-1')
    data['player_playoffs'] = pd.read_csv(DATA_DIR / 'player_playoffs.txt', encoding='latin-1')
    data['player_allstar'] = pd.read_csv(DATA_DIR / 'player_allstar.txt', encoding='latin-1')
    data['team_season'] = pd.read_csv(DATA_DIR / 'team_season.txt', encoding='latin-1')
    data['teams'] = pd.read_csv(DATA_DIR / 'teams.txt', encoding='latin-1')
    # Handle draft.txt with error handling for malformed lines
    data['draft'] = pd.read_csv(DATA_DIR / 'draft.txt', encoding='latin-1', on_bad_lines='skip')
    
    print(f"✓ Loaded {len(data)} datasets")
    return data

def visualize_player_demographics(players_df):
    """Visualize player demographics"""
    print("\n1. Creating player demographics visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Height distribution
    players_df['height_inches'] = players_df['h_feet'] * 12 + players_df['h_inches']
    axes[0, 0].hist(players_df['height_inches'].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Height (inches)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Player Height Distribution')
    
    # Weight distribution
    axes[0, 1].hist(players_df['weight'].dropna(), bins=30, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Weight (lbs)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Player Weight Distribution')
    
    # Position distribution
    position_counts = players_df['position'].value_counts()
    axes[1, 0].bar(position_counts.index, position_counts.values, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Number of Players')
    axes[1, 0].set_title('Players by Position')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Career length
    players_df['career_length'] = players_df['lastseason'] - players_df['firstseason']
    axes[1, 1].hist(players_df['career_length'].dropna(), bins=30, color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('Career Length (years)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Player Career Length Distribution')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'player_demographics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/player_demographics.png")
    plt.close()

def visualize_scoring_trends(player_season_df):
    """Visualize scoring trends over time"""
    print("\n2. Creating scoring trends visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average points per year
    yearly_stats = player_season_df.groupby('year').agg({
        'pts': 'mean',
        'reb': 'mean',
        'asts': 'mean',
        'fgm': 'sum',
        'fga': 'sum'
    }).reset_index()
    
    yearly_stats['fg_pct'] = (yearly_stats['fgm'] / yearly_stats['fga']) * 100
    
    axes[0, 0].plot(yearly_stats['year'], yearly_stats['pts'], linewidth=2, color='crimson')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Average Points')
    axes[0, 0].set_title('Average Points Per Player Per Season (1946-2005)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rebounds trend
    axes[0, 1].plot(yearly_stats['year'], yearly_stats['reb'], linewidth=2, color='darkblue')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Rebounds')
    axes[0, 1].set_title('Average Rebounds Per Player Per Season')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Assists trend
    axes[1, 0].plot(yearly_stats['year'], yearly_stats['asts'], linewidth=2, color='darkgreen')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Average Assists')
    axes[1, 0].set_title('Average Assists Per Player Per Season')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Field goal percentage trend
    axes[1, 1].plot(yearly_stats['year'], yearly_stats['fg_pct'], linewidth=2, color='darkorange')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Field Goal Percentage')
    axes[1, 1].set_title('Average Field Goal Percentage Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scoring_trends.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/scoring_trends.png")
    plt.close()

def visualize_top_performers(player_season_df, players_df):
    """Visualize top performers"""
    print("\n3. Creating top performers visualization...")
    
    # Merge to get player names
    merged_df = player_season_df.merge(
        players_df[['ilkid', 'firstname', 'lastname']], 
        on='ilkid', 
        how='left',
        suffixes=('', '_player')
    )
    merged_df['fullname'] = merged_df['firstname_player'] + ' ' + merged_df['lastname_player']
    
    # Top scorers (single season)
    top_scorers = merged_df.nlargest(15, 'pts')[['fullname', 'pts', 'year']].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top 15 single-season scorers
    axes[0, 0].barh(range(len(top_scorers)), top_scorers['pts'], color='crimson', edgecolor='black')
    axes[0, 0].set_yticks(range(len(top_scorers)))
    axes[0, 0].set_yticklabels([f"{name} ({year})" for name, year in zip(top_scorers['fullname'], top_scorers['year'])], fontsize=8)
    axes[0, 0].set_xlabel('Points')
    axes[0, 0].set_title('Top 15 Single-Season Scoring Performances')
    axes[0, 0].invert_yaxis()
    
    # Top rebounders
    top_rebounders = merged_df.nlargest(15, 'reb')[['fullname', 'reb', 'year']].copy()
    axes[0, 1].barh(range(len(top_rebounders)), top_rebounders['reb'], color='darkblue', edgecolor='black')
    axes[0, 1].set_yticks(range(len(top_rebounders)))
    axes[0, 1].set_yticklabels([f"{name} ({year})" for name, year in zip(top_rebounders['fullname'], top_rebounders['year'])], fontsize=8)
    axes[0, 1].set_xlabel('Rebounds')
    axes[0, 1].set_title('Top 15 Single-Season Rebounding Performances')
    axes[0, 1].invert_yaxis()
    
    # Top assist leaders
    top_assisters = merged_df.nlargest(15, 'asts')[['fullname', 'asts', 'year']].copy()
    axes[1, 0].barh(range(len(top_assisters)), top_assisters['asts'], color='darkgreen', edgecolor='black')
    axes[1, 0].set_yticks(range(len(top_assisters)))
    axes[1, 0].set_yticklabels([f"{name} ({year})" for name, year in zip(top_assisters['fullname'], top_assisters['year'])], fontsize=8)
    axes[1, 0].set_xlabel('Assists')
    axes[1, 0].set_title('Top 15 Single-Season Assist Performances')
    axes[1, 0].invert_yaxis()
    
    # Career total points leaders
    career_points = merged_df.groupby('ilkid').agg({
        'pts': 'sum',
        'fullname': 'first'
    }).nlargest(15, 'pts')
    
    axes[1, 1].barh(range(len(career_points)), career_points['pts'], color='purple', edgecolor='black')
    axes[1, 1].set_yticks(range(len(career_points)))
    axes[1, 1].set_yticklabels(career_points['fullname'], fontsize=8)
    axes[1, 1].set_xlabel('Career Points')
    axes[1, 1].set_title('Top 15 Career Points Leaders')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_performers.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/top_performers.png")
    plt.close()

def visualize_team_performance(team_season_df):
    """Visualize team performance over time"""
    print("\n4. Creating team performance visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Win percentage calculation
    team_season_df['win_pct'] = team_season_df['won'] / (team_season_df['won'] + team_season_df['lost'])
    team_season_df['total_games'] = team_season_df['won'] + team_season_df['lost']
    
    # Average wins per season over time
    yearly_wins = team_season_df.groupby('year')['won'].mean().reset_index()
    axes[0, 0].plot(yearly_wins['year'], yearly_wins['won'], linewidth=2, color='navy')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Average Wins')
    axes[0, 0].set_title('Average Team Wins Per Season')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Points scored vs points allowed
    axes[0, 1].scatter(team_season_df['o_pts'], team_season_df['d_pts'], 
                       c=team_season_df['win_pct'], cmap='RdYlGn', alpha=0.6, s=50)
    axes[0, 1].set_xlabel('Offensive Points')
    axes[0, 1].set_ylabel('Defensive Points (Allowed)')
    axes[0, 1].set_title('Offensive vs Defensive Performance (colored by win %)')
    axes[0, 1].plot([2000, 10000], [2000, 10000], 'k--', alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Win Percentage')
    
    # Top teams by wins
    team_wins = team_season_df.groupby('team')['won'].sum().nlargest(20)
    axes[1, 0].barh(range(len(team_wins)), team_wins.values, color='teal', edgecolor='black')
    axes[1, 0].set_yticks(range(len(team_wins)))
    axes[1, 0].set_yticklabels(team_wins.index, fontsize=9)
    axes[1, 0].set_xlabel('Total Wins')
    axes[1, 0].set_title('Top 20 Teams by Total Wins (All Time)')
    axes[1, 0].invert_yaxis()
    
    # Offensive points trend
    yearly_offense = team_season_df.groupby('year')['o_pts'].mean().reset_index()
    axes[1, 1].plot(yearly_offense['year'], yearly_offense['o_pts'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Average Points Per Team')
    axes[1, 1].set_title('Average Team Offensive Points Per Season')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'team_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/team_performance.png")
    plt.close()

def visualize_draft_analysis(draft_df):
    """Visualize draft analysis"""
    print("\n5. Creating draft analysis visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Filter out invalid data
    draft_clean = draft_df[draft_df['draft_year'] > 1946].copy()
    
    # Drafts per year
    drafts_per_year = draft_clean['draft_year'].value_counts().sort_index()
    axes[0, 0].plot(drafts_per_year.index, drafts_per_year.values, linewidth=2, color='darkviolet')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Draft Picks')
    axes[0, 0].set_title('Number of Players Drafted Per Year')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top colleges by draft picks
    top_colleges = draft_clean['draft_from'].value_counts().head(20)
    axes[0, 1].barh(range(len(top_colleges)), top_colleges.values, color='orange', edgecolor='black')
    axes[0, 1].set_yticks(range(len(top_colleges)))
    axes[0, 1].set_yticklabels(top_colleges.index, fontsize=8)
    axes[0, 1].set_xlabel('Number of Players Drafted')
    axes[0, 1].set_title('Top 20 Colleges by Players Drafted')
    axes[0, 1].invert_yaxis()
    
    # Draft rounds over time
    draft_clean['draft_round'] = pd.to_numeric(draft_clean['draft_round'], errors='coerce')
    rounds_by_year = draft_clean.groupby('draft_year')['draft_round'].max()
    axes[1, 0].plot(rounds_by_year.index, rounds_by_year.values, linewidth=2, color='brown')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Number of Rounds')
    axes[1, 0].set_title('Draft Rounds Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Teams with most draft picks
    top_draft_teams = draft_clean['team'].value_counts().head(20)
    axes[1, 1].barh(range(len(top_draft_teams)), top_draft_teams.values, color='steelblue', edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_draft_teams)))
    axes[1, 1].set_yticklabels(top_draft_teams.index, fontsize=9)
    axes[1, 1].set_xlabel('Number of Draft Picks')
    axes[1, 1].set_title('Top 20 Teams by Total Draft Picks')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'draft_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/draft_analysis.png")
    plt.close()

def visualize_allstar_statistics(allstar_df):
    """Visualize All-Star game statistics"""
    print("\n6. Creating All-Star statistics visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # All-Star appearances by player
    allstar_counts = allstar_df.groupby(['ilkid', 'firstname', 'lastname']).size().reset_index(name='appearances')
    allstar_counts['fullname'] = allstar_counts['firstname'] + ' ' + allstar_counts['lastname']
    top_allstars = allstar_counts.nlargest(20, 'appearances')
    
    axes[0, 0].barh(range(len(top_allstars)), top_allstars['appearances'], color='gold', edgecolor='black')
    axes[0, 0].set_yticks(range(len(top_allstars)))
    axes[0, 0].set_yticklabels(top_allstars['fullname'], fontsize=9)
    axes[0, 0].set_xlabel('Number of All-Star Appearances')
    axes[0, 0].set_title('Top 20 Players by All-Star Appearances')
    axes[0, 0].invert_yaxis()
    
    # Average All-Star points over time
    yearly_allstar = allstar_df.groupby('year')['pts'].mean().reset_index()
    axes[0, 1].plot(yearly_allstar['year'], yearly_allstar['pts'], linewidth=2, color='red', marker='o')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Points')
    axes[0, 1].set_title('Average All-Star Game Points Per Player')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Conference distribution
    conference_counts = allstar_df['conference'].value_counts()
    axes[1, 0].pie(conference_counts.values, labels=conference_counts.index, autopct='%1.1f%%',
                   colors=['lightblue', 'lightcoral'], startangle=90)
    axes[1, 0].set_title('All-Star Selections by Conference')
    
    # Top All-Star scorers (single game)
    top_game_scorers = allstar_df.nlargest(15, 'pts')[['firstname', 'lastname', 'pts', 'year']].copy()
    top_game_scorers['fullname'] = top_game_scorers['firstname'] + ' ' + top_game_scorers['lastname']
    
    axes[1, 1].barh(range(len(top_game_scorers)), top_game_scorers['pts'], color='crimson', edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_game_scorers)))
    axes[1, 1].set_yticklabels([f"{name} ({year})" for name, year in 
                                 zip(top_game_scorers['fullname'], top_game_scorers['year'])], fontsize=8)
    axes[1, 1].set_xlabel('Points')
    axes[1, 1].set_title('Top 15 All-Star Game Scoring Performances')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'allstar_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/allstar_statistics.png")
    plt.close()

def generate_summary_statistics(data):
    """Generate and print summary statistics"""
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\n📊 Players Dataset:")
    print(f"   - Total players: {len(data['players']):,}")
    print(f"   - Average height: {data['players']['height_inches'].mean():.1f} inches")
    print(f"   - Average weight: {data['players']['weight'].mean():.1f} lbs")
    print(f"   - Average career length: {data['players']['career_length'].mean():.1f} years")
    
    print(f"\n📊 Regular Season Dataset:")
    print(f"   - Total player-seasons: {len(data['player_regular_season']):,}")
    print(f"   - Years covered: {data['player_regular_season']['year'].min()} - {data['player_regular_season']['year'].max()}")
    print(f"   - Average points per game: {data['player_regular_season']['pts'].mean():.1f}")
    
    print(f"\n📊 Team Season Dataset:")
    print(f"   - Total team-seasons: {len(data['team_season']):,}")
    print(f"   - Unique teams: {data['team_season']['team'].nunique()}")
    print(f"   - Average team wins: {data['team_season']['won'].mean():.1f}")
    
    print(f"\n📊 Draft Dataset:")
    print(f"   - Total draft picks: {len(data['draft']):,}")
    print(f"   - Years covered: {data['draft']['draft_year'].min()} - {data['draft']['draft_year'].max()}")
    
    print(f"\n📊 All-Star Dataset:")
    print(f"   - Total All-Star appearances: {len(data['player_allstar']):,}")
    print(f"   - Unique players: {data['player_allstar']['ilkid'].nunique()}")
    
    print("\n" + "="*60)

def main():
    """Main function to run all visualizations"""
    print("="*60)
    print("BASKETBALL ASSOCIATION DATA VISUALIZATION")
    print("="*60)
    
    # Load data
    data = load_data()
    
    # Generate visualizations
    visualize_player_demographics(data['players'])
    visualize_scoring_trends(data['player_regular_season'])
    visualize_top_performers(data['player_regular_season'], data['players'])
    visualize_team_performance(data['team_season'])
    visualize_draft_analysis(data['draft'])
    visualize_allstar_statistics(data['player_allstar'])
    
    # Generate summary statistics
    generate_summary_statistics(data)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files in output/ folder:")
    print("  1. player_demographics.png")
    print("  2. scoring_trends.png")
    print("  3. top_performers.png")
    print("  4. team_performance.png")
    print("  5. draft_analysis.png")
    print("  6. allstar_statistics.png")
    print("\n")

if __name__ == "__main__":
    main()
