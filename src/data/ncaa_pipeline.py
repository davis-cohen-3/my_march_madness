import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import random

class NCAADataPipeline:
    def __init__(self, data_dir: str = "data"):
        """Initialize the NCAA data pipeline with directory structure."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw/mens"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
        
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
            'regular_season': pd.read_csv(self.raw_dir / "MRegularSeasonDetailedResults.csv"),
            'tourney': pd.read_csv(self.raw_dir / "MNCAATourneyDetailedResults.csv"),
            'massey': pd.read_csv(self.raw_dir / "MMasseyOrdinals_thruSeason2024_day128.csv"),
            'seeds': pd.read_csv(self.raw_dir / "MNCAATourneySeeds.csv")
        }
        
        # Filter for years we want
        for key in ['regular_season', 'tourney']:
            data_files[key] = data_files[key][data_files[key]['Season'] >= start_year]
            
        return data_files
    
    def safe_ratio(self, a, b):
        if b == 0:
            return 0
        return a / b

    def calculate_regular_season_stats(self, regular_season_df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Calculate team statistics using ONLY regular season games."""
        try:
            total_teams = regular_season_df['WTeamID'].nunique() + regular_season_df['LTeamID'].nunique()
            self.logger.info(f"Processing {total_teams} unique teams for season {season}")

            season_games = regular_season_df[regular_season_df['Season'] == season].copy()
            
            # Add validation for required columns
            required_columns = {
                'win_cols': ['WTeamID', 'WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 
                            'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk'],
                'loss_cols': ['LTeamID', 'LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 
                            'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk']
            }
            
            # Check for missing columns
            missing_cols = []
            for col_group in required_columns.values():
                missing = [col for col in col_group if col not in season_games.columns]
                if missing:
                    missing_cols.extend(missing)
                    
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()  # Return empty DataFrame if missing required columns
                
            # Create team perspective for wins with null checking
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
            
            # Create team perspective for losses with null checking
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
            min_games_required = 5  # Adjust this threshold as needed
            valid_teams = games_per_team[games_per_team >= min_games_required].index
            
            if len(valid_teams) < len(games_per_team):
                self.logger.warning(
                    f"Removed {len(games_per_team) - len(valid_teams)} teams with fewer than {min_games_required} games"
                )
            
            all_games = all_games[all_games['TeamID'].isin(valid_teams)]
            
            # Calculate regular season stats with null checking
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
            
            # Add data quality flags
            stats['data_quality'] = 'valid'  # Default flag
            
            # Calculate additional metrics with quality flags
            def safe_calc(row, metric_name, calc_func):
                """Safely calculate a metric and update data quality if calculation fails."""
                try:
                    value = calc_func(row)
                    if pd.isna(value) or np.isinf(value):
                        row['data_quality'] = f'invalid_{metric_name}'
                        return None
                    return value.round(3)
                except Exception as e:
                    row['data_quality'] = f'invalid_{metric_name}'
                    return None
            
            # Apply calculations with quality checks
            for idx in stats.index:
                row = stats.loc[idx]
                
                stats.at[idx, 'FGPct'] = safe_calc(row, 'FGPct', 
                    lambda r: self.safe_ratio(r['FGM_mean'], r['FGA_mean']))
                
                stats.at[idx, 'FG3Pct'] = safe_calc(row, 'FG3Pct',
                    lambda r: self.safe_ratio(r['FGM3_mean'], r['FGA3_mean']))
                
                stats.at[idx, 'FTPct'] = safe_calc(row, 'FTPct',
                    lambda r: self.safe_ratio(r['FTM_mean'], r['FTA_mean']))
                
                stats.at[idx, 'RebMargin'] = safe_calc(row, 'RebMargin',
                    lambda r: r['OR_mean'] + r['DR_mean'])
                
                stats.at[idx, 'AssistToTurnover'] = safe_calc(row, 'AssistToTurnover',
                    lambda r: self.safe_ratio(r['Ast_mean'], r['TO_mean']))
                
                stats.at[idx, 'ScoringMargin'] = safe_calc(row, 'ScoringMargin',
                    lambda r: r['PointsScored_mean'] - r['PointsAllowed_mean'])
            
            # Filter out teams with any invalid calculations
            valid_stats = stats[stats['data_quality'] == 'valid']
            
            if len(valid_stats) < len(stats):
                self.logger.warning(
                    f"Removed {len(stats) - len(valid_stats)} teams due to invalid calculations"
                )
                
            return valid_stats
            
        except Exception as e:
            self.logger.error(f'Issue in calculate_regular_season_stats: {str(e)}')
            return pd.DataFrame()
    
    def extract_seed_number(self, seed_string: str) -> int:
        """Extract numeric seed value from seed string (e.g., 'W01', 'X11', 'Y16a', 'Z16b')."""
        # Remove the region prefix (first character)
        seed_without_region = seed_string[1:]
        
        # Extract just the numeric part using string methods
        # This will handle cases like '16a' or '16b' by removing the trailing letter
        numeric_part = ''.join(c for c in seed_without_region if c.isdigit())
        
        return int(numeric_part)
        

    def create_tournament_matchups(self, 
                             tourney_games: pd.DataFrame, 
                             regular_season_stats: pd.DataFrame,
                             seeds_df: pd.DataFrame,
                             random_seed: int = 42) -> pd.DataFrame:
        """Create tournament matchup dataset using regular season statistics."""

        if not isinstance(tourney_games, pd.DataFrame):
            raise TypeError("tourney_games must be a pandas DataFrame")
        if not isinstance(regular_season_stats, pd.DataFrame):
            raise TypeError("regular_season_stats must be a pandas DataFrame")
        
        random.seed(random_seed)
        matchups = []

        # Define required statistics
        required_stats = [
            'WinPct', 'PointsScored_mean', 'PointsAllowed_mean', 'ScoringMargin',
            'FGPct', 'FG3Pct', 'FTPct', 'RebMargin', 'AssistToTurnover',
            'PointsScored_std', 'PointsAllowed_std'
        ]
        
        for _, game in tourney_games.iterrows():
            try:
                # Get tournament seeds for both teams
                season_seeds = seeds_df[seeds_df['Season'] == game['Season']]
                w_seed = season_seeds[season_seeds['TeamID'] == game['WTeamID']]['Seed'].iloc[0]
                l_seed = season_seeds[season_seeds['TeamID'] == game['LTeamID']]['Seed'].iloc[0]
                
                # Randomly decide team ordering
                if random.random() > 0.5:
                    team_a_id = game['WTeamID']
                    team_b_id = game['LTeamID']
                    team_a_seed = w_seed
                    team_b_seed = l_seed
                    target = 1  # Team A wins
                else:
                    team_a_id = game['LTeamID']
                    team_b_id = game['WTeamID']
                    team_a_seed = l_seed
                    team_b_seed = w_seed
                    target = 0  # Team B wins
                    
                # Get regular season stats for both teams
                team_a_stats = regular_season_stats.loc[team_a_id]
                team_b_stats = regular_season_stats.loc[team_b_id]
                
                # Check if all required stats are available
                if not all(stat in team_a_stats.index for stat in required_stats):
                    self.logger.warning(f"Missing required statistics for team {team_a_id} in season {game['Season']}")
                    continue
                if not all(stat in team_b_stats.index for stat in required_stats):
                    self.logger.warning(f"Missing required statistics for team {team_b_id} in season {game['Season']}")
                    continue
                
                # Extract seed numbers (removing region letter)
                team_a_seed_num = self.extract_seed_number(team_a_seed)
                team_b_seed_num = self.extract_seed_number(team_b_seed)
                
                # Calculate matchup features
                features = {
                    'Season': game['Season'],
                    'TeamA_ID': team_a_id,
                    'TeamB_ID': team_b_id,
                    'TeamA_Seed': team_a_seed_num,
                    'TeamB_Seed': team_b_seed_num,
                    'SeedDiff': team_b_seed_num - team_a_seed_num,
                    
                    # Regular season performance differentials
                    'WinPct_diff': team_a_stats['WinPct'] - team_b_stats['WinPct'],
                    'ScoreMargin_diff': team_a_stats['ScoringMargin'] - team_b_stats['ScoringMargin'],
                    'PointsScored_diff': team_a_stats['PointsScored_mean'] - team_b_stats['PointsScored_mean'],
                    'PointsAllowed_diff': team_a_stats['PointsAllowed_mean'] - team_b_stats['PointsAllowed_mean'],
                    
                    # Shooting efficiency ratios
                    'FGPct_ratio': team_a_stats['FGPct'] / team_b_stats['FGPct'],
                    'FG3Pct_ratio': team_a_stats['FG3Pct'] / team_b_stats['FG3Pct'],
                    'FTPct_ratio': team_a_stats['FTPct'] / team_b_stats['FTPct'],
                    
                    # Other performance ratios
                    'RebMargin_ratio': team_a_stats['RebMargin'] / team_b_stats['RebMargin'],
                    'AssistToTurnover_ratio': team_a_stats['AssistToTurnover'] / team_b_stats['AssistToTurnover'],
                    
                    # Performance consistency differences
                    'Scoring_std_diff': team_a_stats['PointsScored_std'] - team_b_stats['PointsScored_std'],
                    'Defense_std_diff': team_a_stats['PointsAllowed_std'] - team_b_stats['PointsAllowed_std'],
                    
                    'Target': target
                }
                
                matchups.append(features)
                
            except Exception as e:
                self.logger.warning(f"Error processing game in season {game['Season']}: {str(e)}")
                continue
        
        if not matchups:
            self.logger.warning("No valid matchups were created!")
            return pd.DataFrame()
            
        return pd.DataFrame(matchups)
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Validate loaded data for completeness and correctness."""
        required_columns = {
            'regular_season': ['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore'],
            'tourney': ['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore'],
            'seeds': ['Season', 'TeamID', 'Seed'],
        }
        
        for key, required_cols in required_columns.items():
            if key not in data:
                raise ValueError(f"Missing required dataset: {key}")
            missing_cols = set(required_cols) - set(data[key].columns)
            if missing_cols:
                raise ValueError(f"Missing required columns in {key}: {missing_cols}")

    def run_pipeline(self, start_year: int = 2003) -> pd.DataFrame:
        """Execute the full pipeline to create the training dataset."""
        # Process in chunks to manage memory
        chunk_size = 1000
        all_matchups = []
        
        # Load raw data
        data = self.load_raw_data(start_year)
        self.validate_data(data)  # Add validation
        
        # Process each season
        for season in sorted(data['tourney']['Season'].unique()):
            self.logger.info(f"Processing season {season}")
            
            # Free memory after each season
            if 'regular_season_stats' in locals():
                del regular_season_stats
            
            # Calculate regular season stats
            regular_season_stats = self.calculate_regular_season_stats(
                data['regular_season'], 
                season
            )
            
            # Create tournament matchups using regular season stats
            tourney_matchups = self.create_tournament_matchups(
                data['tourney'][data['tourney']['Season'] == season],
                regular_season_stats,
                data['seeds']
            )
            
            all_matchups.append(tourney_matchups)
        
        final_dataset = pd.concat(all_matchups, ignore_index=True)
        
        # Save processed dataset
        output_path = self.processed_dir / "tournament_matchups.csv"
        final_dataset.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed dataset to {output_path}")
        
        # Print dataset information
        self.logger.info("\nDataset Summary:")
        self.logger.info(f"Shape: {final_dataset.shape}")
        self.logger.info("\nClass balance:")
        self.logger.info(final_dataset['Target'].value_counts(normalize=True))
        
        return final_dataset

if __name__ == "__main__":
    pipeline = NCAADataPipeline()
    dataset = pipeline.run_pipeline()