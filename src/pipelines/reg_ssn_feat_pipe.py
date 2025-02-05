import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

class RegularSeasonBoxScoreStatsPipeline:

    def __init__(self, data_dir: str = "data"):
        """Initialize the Basic Stats feature engineering pipeline with directory structure."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw/mens"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"

        self.features_df = pd.DataFrame()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self, start_year: int = 2003) -> Dict[str, pd.DataFrame]:
        """Load all necessary raw data files."""
        self.logger.info(f"Loading raw data files from {start_year} onwards...")
        
        data_files = {
            'regular_season': pd.read_csv(self.raw_dir / "MRegularSeasonDetailedResults.csv")
        }
        
        return data_files
    
    def safe_calc(row, metric_name, calc_func):
        """
        Safely calculate a metric without modifying the input row.
        Returns tuple of (calculated_value, data_quality_flag).
        """
        try:
            value = calc_func(row)
            if pd.isna(value) or np.isinf(value):
                return None, f'invalid_{metric_name}'
            return value.round(3), 'valid'
        except Exception as e:
            return None, f'invalid_{metric_name}'
        
    def safe_ratio(self, a, b):
        """
        Safely performs division , taking care of cases where we divide by zero 
        """
        if b == 0:
            return 0
        return a / b
    
    def create_team_season_pairs(self, start_year: int = 2003, end_year: int = 2024) -> pd.DataFrame:
        """Create base DataFrame with unique team-season pairs."""
        self.logger.info(f"Creating team-season pairs from {start_year} to {end_year}")
        
        # Create all possible season-team combinations
        seasons = range(start_year, end_year + 1)
        teams = pd.read_csv(self.raw_dir / "MTeams.csv")
        
        # Filter teams based on FirstD1Season and LastD1Season
        valid_teams = []
        for _, team in teams.iterrows():
            team_seasons = []
            for season in seasons:
                if season >= team['FirstD1Season'] and season <= team['LastD1Season']:
                    team_seasons.append({
                        'season': season,
                        'team_id': team['TeamID']
                    })
            valid_teams.extend(team_seasons)
        
        features_df = pd.DataFrame(valid_teams)
        return features_df
    
    def add_regular_season_features(self, features_df: pd.DataFrame, regular_season_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regular season statistics to the features DataFrame with robust error handling and data validation.
        """
        try:
            self.logger.info("Adding regular season statistics...")
            features = features_df.copy()
            
            # Required columns validation
            required_columns = {
                'win_cols': ['WTeamID', 'WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 
                            'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk'],
                'loss_cols': ['LTeamID', 'LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 
                            'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk']
            }
            
            # Process each season
            all_season_stats = []
            for season in features['season'].unique():
                try:
                    self.logger.info(f"Processing regular season stats for {season}")
                    
                    # Filter games for this season
                    season_games = regular_season_df[regular_season_df['Season'] == season].copy()
                    
                    # Create team perspective for wins
                    win_perspective = season_games[required_columns['win_cols']].rename(columns={
                        'WTeamID': 'TeamID',
                        'WScore': 'PointsScored',
                        'LScore': 'PointsAllowed',
                        'WFGM': 'FGM', 'WFGA': 'FGA',
                        'WFGM3': 'FGM3', 'WFGA3': 'FGA3',
                        'WFTM': 'FTM', 'WFTA': 'FTA',
                        'WOR': 'OR', 'WDR': 'DR',
                        'WAst': 'Ast', 'WTO': 'TO',
                        'WStl': 'Stl', 'WBlk': 'Blk'
                    })
                    win_perspective['Won'] = 1
                    
                    # Create team perspective for losses
                    loss_perspective = season_games[required_columns['loss_cols']].rename(columns={
                        'LTeamID': 'TeamID',
                        'LScore': 'PointsScored',
                        'WScore': 'PointsAllowed',
                        'LFGM': 'FGM', 'LFGA': 'FGA',
                        'LFGM3': 'FGM3', 'LFGA3': 'FGA3',
                        'LFTM': 'FTM', 'LFTA': 'FTA',
                        'LOR': 'OR', 'LDR': 'DR',
                        'LAst': 'Ast', 'LTO': 'TO',
                        'LStl': 'Stl', 'LBlk': 'Blk'
                    })
                    loss_perspective['Won'] = 0
                    
                    # Combine all regular season games
                    all_games = pd.concat([win_perspective, loss_perspective])
                    
                    # Check for teams with insufficient data
                    games_per_team = all_games.groupby('TeamID').size()
                    min_games_required = 5
                    valid_teams = games_per_team[games_per_team >= min_games_required].index
                    
                    if len(valid_teams) < len(games_per_team):
                        self.logger.warning(
                            f"Season {season}: Removed {len(games_per_team) - len(valid_teams)} teams " +
                            f"with fewer than {min_games_required} games"
                        )
                    
                    all_games = all_games[all_games['TeamID'].isin(valid_teams)]
                    
                    # Calculate season stats
                    stats = all_games.groupby('TeamID').agg({
                        'Won': ['count', 'sum', 'mean'],
                        'PointsScored': ['mean', 'std'],
                        'PointsAllowed': ['mean', 'std'],
                        'FGM': 'mean',
                        'FGA': 'mean',
                        'FGM3': 'mean',
                        'FGA3': 'mean',
                        'FTM': 'mean',
                        'FTA': 'mean',
                        'OR': 'mean',
                        'DR': 'mean',
                        'Ast': 'mean',
                        'TO': 'mean',
                        'Stl': 'mean',
                        'Blk': 'mean'
                    })
                    
                    # Flatten column names
                    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
                    stats = stats.rename(columns={
                        'Won_count': 'Games',
                        'Won_sum': 'Wins',
                        'Won_mean': 'WinPct'
                    })
                    
                    # Initialize data quality column
                    stats['data_quality'] = 'valid'
                    
                    # Calculate additional metrics with quality checks
                    derived_metrics = {}
                    for idx in stats.index:
                        row = stats.loc[idx]
                        
                        # Calculate metrics using safe_ratio
                        fg_pct = self.safe_ratio(row['FGM_mean'], row['FGA_mean'])
                        fg3_pct = self.safe_ratio(row['FGM3_mean'], row['FGA3_mean'])
                        ft_pct = self.safe_ratio(row['FTM_mean'], row['FTA_mean'])
                        ast_to = self.safe_ratio(row['Ast_mean'], row['TO_mean'])
                        reb_margin = row['OR_mean'] + row['DR_mean']
                        score_margin = row['PointsScored_mean'] - row['PointsAllowed_mean']
                        
                        derived_metrics[idx] = {
                            'FGPct': fg_pct,
                            'FG3Pct': fg3_pct,
                            'FTPct': ft_pct,
                            'RebMargin': reb_margin, #should be rebounds/game
                            'AssistToTurnover': ast_to,
                            'ScoringMargin': score_margin
                        }
                    
                    # Add derived metrics to stats DataFrame
                    derived_df = pd.DataFrame.from_dict(derived_metrics, orient='index')
                    stats = stats.join(derived_df)
                    
                    # Add season column and reset index
                    stats['season'] = season
                    stats = stats.reset_index().rename(columns={'TeamID': 'team_id'})
                    
                    all_season_stats.append(stats)
                    
                except Exception as e:
                    self.logger.error(f"Error processing season {season}: {str(e)}")
                    continue
            
            if not all_season_stats:
                self.logger.error("No valid season statistics were calculated")
                return features
            
            # Combine all seasons
            all_stats = pd.concat(all_season_stats, ignore_index=True)
            
            # Merge with features DataFrame
            features = features.merge(
                all_stats,
                on=['season', 'team_id'],
                how='left',
                validate='1:1'  # Ensure one-to-one merge
            )
            
            # Fill missing values with appropriate defaults
            fill_values = {
                'Games': 0,
                'Wins': 0,
                'WinPct': 0,
                'PointsScored_mean': 0,
                'PointsScored_std': 0,
                'PointsAllowed_mean': 0,
                'PointsAllowed_std': 0,
                'FGPct': 0,
                'FG3Pct': 0,
                'FTPct': 0,
                'RebMargin': 0,
                'AssistToTurnover': 0,
                'ScoringMargin': 0,
                'data_quality': 'missing'
            }
            
            features = features.fillna(fill_values)

            # self.logger.info("\nAdvanced Stats Summary:")
            # for col in new_columns:
            #     stats = features[col].describe()
            #     self.logger.info(f"\n{col}:")
            #     self.logger.info(f"Mean: {stats['mean']:.2f}")
            #     self.logger.info(f"Std: {stats['std']:.2f}")
            #     self.logger.info(f"Min: {stats['min']:.2f}")
            #     self.logger.info(f"Max: {stats['max']:.2f}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Critical error in add_regular_season_features: {str(e)}")

    def add_advanced_stats(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced basketball statistics to the features DataFrame.
        Calculates: eFG%, Offensive Rating, Defensive Rating, Net Rating, Pace, Assist Percentage
        """
        try:
            self.logger.info("Adding advanced basketball statistics...")
            features = features_df.copy()
            
            def calculate_possessions(row):
                """Calculate possessions using the advanced formula with safe division"""
                try:
                    # Safe division for rebound percentage
                    total_rebounds = row['OR_mean'] + row['DR_mean']
                    rebound_pct = row['OR_mean'] / total_rebounds if total_rebounds > 0 else 0
                    
                    # Team possessions
                    team_poss = (
                        row['FGA_mean'] + 0.4 * row['FTA_mean'] - 
                        1.07 * rebound_pct * 
                        (row['FGA_mean'] - row['FGM_mean']) + row['TO_mean']
                    )
                    
                    # Safe division for opponent possessions estimation
                    opp_poss = (team_poss * (row['PointsAllowed_mean'] / row['PointsScored_mean']) 
                            if row['PointsScored_mean'] > 0 else team_poss)
                    
                    return 0.5 * (team_poss + opp_poss)
                except Exception as e:
                    # Log error and return a default value
                    self.logger.warning(f"Error calculating possessions: {str(e)}")
                    return 0
            
            # Calculate possessions per game
            features['Possessions'] = features.apply(calculate_possessions, axis=1)
            
            # Calculate Pace (possessions per game, normalized to 40 minutes)
            features['Pace'] = features['Possessions'] * (40/40)  # Adjust multiplier if games aren't 40 minutes
            
            # Calculate Effective Field Goal Percentage
            features['eFG_Pct'] = features.apply(
                lambda row: (row['FGM_mean'] + 0.5 * row['FGM3_mean']) / row['FGA_mean'] 
                if row['FGA_mean'] > 0 else 0, 
                axis=1
            )
            
            # Calculate Offensive Rating (Points per 100 possessions)
            features['OffRating'] = features.apply(
                lambda row: (row['PointsScored_mean'] * 100) / row['Possessions'] 
                if row['Possessions'] > 0 else 0,
                axis=1
            )
            
            # Calculate Defensive Rating (Points allowed per 100 possessions)
            features['DefRating'] = features.apply(
                lambda row: (row['PointsAllowed_mean'] * 100) / row['Possessions']
                if row['Possessions'] > 0 else 0,
                axis=1
            )
            
            # Calculate Net Rating
            features['NetRating'] = features['OffRating'] - features['DefRating']
            
            # Calculate Assist Percentage
            features['AssistPct'] = features.apply(
                lambda row: (row['Ast_mean'] * 100) / row['FGM_mean']
                if row['FGM_mean'] > 0 else 0,
                axis=1
            )
            
            # Round all new statistics to 2 decimal places
            new_columns = ['Pace', 'eFG_Pct', 'OffRating', 'DefRating', 'NetRating', 'AssistPct']
            features[new_columns] = features[new_columns].round(2)
            
            # Add data quality checks
            features['advanced_stats_quality'] = 'valid'
            
            # Mark records with potentially problematic values
            quality_conditions = [
                (features['Pace'] <= 0, 'invalid_pace'),
                (features['eFG_Pct'] > 1, 'invalid_efg'),
                (features['OffRating'] <= 0, 'invalid_offrating'),
                (features['DefRating'] <= 0, 'invalid_defrating'),
                (features['AssistPct'] > 100, 'invalid_assistpct')
            ]
            
            for condition, flag in quality_conditions:
                features.loc[condition, 'advanced_stats_quality'] = flag
                
            # Fill any NaN values with 0
            features[new_columns] = features[new_columns].fillna(0)
            
            self.logger.info("Advanced statistics successfully added")
            
            # Log some summary statistics
            self.logger.info("\nAdvanced Stats Summary:")
            for col in new_columns:
                stats = features[col].describe()
                self.logger.info(f"\n{col}:")
                self.logger.info(f"Mean: {stats['mean']:.2f}")
                self.logger.info(f"Std: {stats['std']:.2f}")
                self.logger.info(f"Min: {stats['min']:.2f}")
                self.logger.info(f"Max: {stats['max']:.2f}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in add_advanced_stats: {str(e)}")
            return features_df

    def run_pipeline(self, start_year: int = 2003, end_year: int = 2024) -> pd.DataFrame:
        """Execute the full pipeline to create the team features dataset."""
        # Load raw data
        data = self.load_raw_data(start_year)
        
        # Create base features DataFrame with team-season pairs
        features_df = self.create_team_season_pairs(start_year, end_year)
        
        # Add Regular Season Stats to features_df
        features_df = self.add_regular_season_features(features_df, data['regular_season'])

        # Add advanced statistics
        features_df = self.add_advanced_stats(features_df)
        
        # Save processed dataset
        output_path = self.features_dir / "regular_season_stats.csv"
        features_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved regular season features dataset to {output_path}")
        
        # Print dataset information
        self.logger.info("\nDataset Summary:")
        self.logger.info(f"Shape: {features_df.shape}")
        self.logger.info("\nFeatures included:")
        self.logger.info(f"Columns: {features_df.columns.tolist()}")
        
        return features_df
    


if __name__ == "__main__":
    pipeline = RegularSeasonBoxScoreStatsPipeline()
    dataset = pipeline.run_pipeline()