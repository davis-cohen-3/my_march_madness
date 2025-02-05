import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional

class ConferencesFeaturePipeline:
    def __init__(self, data_dir: str = "data"):
        """Initialize the Conference feature engineering pipeline."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw/mens"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Tournament round points
        self.round_points = {
            'First Four': 1,  # Play-in games
            'R64': 1,
            'R32': 2,
            'S16': 3,
            'E8': 4,
            'F4': 5,
            'NCG': 6
        }
        
        # Ranking tiers points
        self.ranking_tiers = {
            'top_5': 4,
            'top_15': 3,
            'top_25': 2,
            'top_35': 1
        }

    def load_data(self, start_year: int = 2003) -> Dict[str, pd.DataFrame]:
        """Load necessary data files."""
        self.logger.info(f"Loading data files from {start_year} onwards...")
        
        data_files = {
            'tournament': pd.read_csv(self.raw_dir / "MNCAATourneyCompactResults.csv"),
            'rankings': pd.read_csv(self.raw_dir / "MMasseyOrdinals_thruSeason2024_day128.csv"),
            'conferences': pd.read_csv(self.raw_dir / "MTeamConferences.csv")
        }
        
        # Filter for years we want
        for key in ['tournament', 'conferences']:
            data_files[key] = data_files[key][data_files[key]['Season'] >= start_year]
            
        return data_files

    def get_tournament_round(self, day_num: int) -> str:
        """Determine tournament round based on DayNum."""
        if day_num in [134, 135]:
            return 'First Four'
        elif day_num in [136, 137]:
            return 'R64'
        elif day_num in [138, 139]:
            return 'R32'
        elif day_num in [143, 144]:
            return 'S16'
        elif day_num in [145, 146]:
            return 'E8'
        elif day_num == 152:
            return 'F4'
        elif day_num == 154:
            return 'NCG'
        else:
            return 'Unknown'

    def calculate_tournament_success(self, tournament_data: pd.DataFrame, 
                                  conference_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical tournament success metrics for each conference by season."""
        self.logger.info("Calculating tournament success metrics...")
        
        success_metrics = []
        
        # Process each season (2003-2024)
        for season in range(2003, 2025):
            # Get previous seasons' tournament data
            historical_games = tournament_data[tournament_data['Season'] < season].copy()
            current_conf_mapping = conference_data[conference_data['Season'] == season]
            
            if len(historical_games) == 0:
                continue
                
            # Calculate points and win percentage for each team
            team_stats = {}
            for _, game in historical_games.iterrows():
                round_name = self.get_tournament_round(game['DayNum'])
                points = self.round_points.get(round_name, 0)
                
                # Update winning team stats
                if game['WTeamID'] not in team_stats:
                    team_stats[game['WTeamID']] = {'points': 0, 'wins': 0, 'games': 0}
                team_stats[game['WTeamID']]['points'] += points
                team_stats[game['WTeamID']]['wins'] += 1
                team_stats[game['WTeamID']]['games'] += 1
                
                # Update losing team stats
                if game['LTeamID'] not in team_stats:
                    team_stats[game['LTeamID']] = {'points': 0, 'wins': 0, 'games': 0}
                team_stats[game['LTeamID']]['games'] += 1
            
            # Aggregate by conference
            conference_stats = {}
            for _, row in current_conf_mapping.iterrows():
                conf = row['ConfAbbrev']
                team_id = row['TeamID']
                
                if conf not in conference_stats:
                    conference_stats[conf] = {
                        'total_points': 0,
                        'total_wins': 0,
                        'total_games': 0,
                        'team_count': 0
                    }
                
                conference_stats[conf]['team_count'] += 1
                if team_id in team_stats:
                    conference_stats[conf]['total_points'] += team_stats[team_id]['points']
                    conference_stats[conf]['total_wins'] += team_stats[team_id]['wins']
                    conference_stats[conf]['total_games'] += team_stats[team_id]['games']
            
            # Calculate normalized metrics
            for conf, stats in conference_stats.items():
                normalized_points = stats['total_points'] / stats['team_count'] if stats['team_count'] > 0 else 0
                win_pct = stats['total_wins'] / stats['total_games'] if stats['total_games'] > 0 else 0
                
                success_metrics.append({
                    'Season': season,
                    'ConfAbbrev': conf,
                    'normalized_historical_tourney_points': round(normalized_points, 3),
                    'historical_tourney_win_pct': round(win_pct, 3)
                })
        
        return pd.DataFrame(success_metrics)

    def calculate_ranking_strength(self, rankings_data: pd.DataFrame, 
                                conference_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ranking-based conference strength metrics."""
        self.logger.info("Calculating ranking-based conference strength...")
        
        strength_metrics = []
        
        # Process each season
        for season in range(2003, 2025):
            season_rankings = rankings_data[rankings_data['Season'] == season]
            current_conf_mapping = conference_data[conference_data['Season'] == season]
            
            if len(season_rankings) == 0:
                continue
            
            # Get final rankings for the season
            max_day = season_rankings['RankingDayNum'].max()
            final_rankings = season_rankings[season_rankings['RankingDayNum'] == max_day]
            
            # Calculate average ranking points for each team
            team_points = {}
            for team_id in final_rankings['TeamID'].unique():
                team_rankings = final_rankings[final_rankings['TeamID'] == team_id]
                
                # Calculate points based on average rank across all systems
                avg_rank = team_rankings['OrdinalRank'].mean()
                
                # Assign points based on tiers
                if avg_rank <= 5:
                    points = self.ranking_tiers['top_5']
                elif avg_rank <= 15:
                    points = self.ranking_tiers['top_15']
                elif avg_rank <= 25:
                    points = self.ranking_tiers['top_25']
                elif avg_rank <= 35:
                    points = self.ranking_tiers['top_35']
                else:
                    points = 0
                    
                team_points[team_id] = points
            
            # Aggregate by conference
            conference_points = {}
            for _, row in current_conf_mapping.iterrows():
                conf = row['ConfAbbrev']
                team_id = row['TeamID']
                
                if conf not in conference_points:
                    conference_points[conf] = {
                        'total_points': 0,
                        'team_count': 0
                    }
                
                conference_points[conf]['team_count'] += 1
                if team_id in team_points:
                    conference_points[conf]['total_points'] += team_points[team_id]
            
            # Calculate normalized strength
            for conf, stats in conference_points.items():
                normalized_strength = stats['total_points'] / stats['team_count'] if stats['team_count'] > 0 else 0
                
                strength_metrics.append({
                    'Season': season,
                    'ConfAbbrev': conf,
                    'normalized_ranking_strength': round(normalized_strength, 3)
                })
        
        return pd.DataFrame(strength_metrics)

    def run_pipeline(self, start_year: int = 2003) -> pd.DataFrame:
        """Execute the full pipeline to create conference features."""
        # Load data
        data = self.load_data(start_year)
        
        # Calculate tournament success metrics
        tournament_metrics = self.calculate_tournament_success(
            data['tournament'],
            data['conferences']
        )
        
        # Calculate ranking strength metrics
        ranking_metrics = self.calculate_ranking_strength(
            data['rankings'],
            data['conferences']
        )
        
        # Merge metrics
        conference_features = tournament_metrics.merge(
            ranking_metrics,
            on=['Season', 'ConfAbbrev'],
            how='outer'
        )
        
        # Sort and clean
        conference_features = conference_features.sort_values(['Season', 'ConfAbbrev'])
        conference_features = conference_features.fillna(0)
        
        # Save to CSV
        output_path = self.features_dir / "conference_features.csv"
        conference_features.to_csv(output_path, index=False)
        self.logger.info(f"Saved conference features to {output_path}")
        
        # Print summary statistics
        self.logger.info("\nFeatures Summary:")
        self.logger.info(f"Total Seasons: {conference_features['Season'].nunique()}")
        self.logger.info(f"Total Conferences: {conference_features['ConfAbbrev'].nunique()}")
        
        return conference_features

if __name__ == "__main__":
    pipeline = ConferencesFeaturePipeline()
    conference_features = pipeline.run_pipeline()