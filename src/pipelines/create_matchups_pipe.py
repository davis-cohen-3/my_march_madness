# import pandas as pd
# import numpy as np
# from pathlib import Path
# from typing import Dict, List, Optional
# from datetime import datetime # optional
# import logging
# import random
# import requests # optional
# from functools import lru_cache # optional

# class FinalPipeline:

#     def __init__(self, data_dir: str = "data"):
#         """Initialize the Enhanced feature engineering pipeline with directory structure."""
#         self.data_dir = Path(data_dir)
#         self.raw_dir = self.data_dir / "raw/mens"
#         self.processed_dir = self.data_dir / "processed"
#         self.features_dir = self.data_dir / "features"

#         self.features_df = pd.DataFrame()
        

#         # Setup logging
#         logging.basicConfig(level=logging.INFO)
#         self.logger = logging.getLogger(__name__)
        
#         # Create directories if they don't exist
#         for dir_path in [self.raw_dir, self.processed_dir, self.features_dir]:
#             dir_path.mkdir(parents=True, exist_ok=True)

#     def merge_all_feature(self) -> int:
#         """Merge all features to store a dataframe representing each team from each season's individual features"""


#     def extract_seed_number(self, seed_string: str) -> int:
#         """Extract numeric seed value from seed string (e.g., 'W01', 'X11', 'Y16a', 'Z16b')."""
#         # Remove the region prefix (first character)
#         seed_without_region = seed_string[1:]
        
#         # Extract just the numeric part using string methods
#         # This will handle cases like '16a' or '16b' by removing the trailing letter
#         numeric_part = ''.join(c for c in seed_without_region if c.isdigit())
        
#         return int(numeric_part)
    
#     def create_tournament_matchups(self, 
#                              tourney_games: pd.DataFrame, 
#                              features: pd.DataFrame,
#                              seeds_df: pd.DataFrame,
#                              random_seed: int = 42) -> pd.DataFrame:
#         """Create tournament matchup dataset using regular season statistics."""

#         if not isinstance(tourney_games, pd.DataFrame):
#             raise TypeError("tourney_games must be a pandas DataFrame")
#         if not isinstance(features, pd.DataFrame):
#             raise TypeError("features_df must be a pandas DataFrame")
        
#         random.seed(random_seed)
#         matchups = []

#         for _, game in tourney_games.iterrows():
#             try:
#                 # Get tournament seeds for both teams
#                 season_seeds = seeds_df[seeds_df['Season'] == game['Season']]
#                 w_seed = season_seeds[season_seeds['TeamID'] == game['WTeamID']]['Seed'].iloc[0]
#                 l_seed = season_seeds[season_seeds['TeamID'] == game['LTeamID']]['Seed'].iloc[0]
                
#                 # Randomly decide team ordering
#                 if random.random() > 0.5:
#                     team_a_id = game['WTeamID']
#                     team_b_id = game['LTeamID']
#                     team_a_seed = w_seed
#                     team_b_seed = l_seed
#                     target = 1  # Team A wins
#                 else:
#                     team_a_id = game['LTeamID']
#                     team_b_id = game['WTeamID']
#                     team_a_seed = l_seed
#                     team_b_seed = w_seed
#                     target = 0  # Team B wins

#                 # Extract seed numbers (removing region letter)
#                 team_a_seed_num = self.extract_seed_number(team_a_seed)
#                 team_b_seed_num = self.extract_seed_number(team_b_seed)

#                 # Calculate matchup features
#                 features = {
#                     'Season': game['Season'],
#                     'TeamA_ID': team_a_id,
#                     'TeamB_ID': team_b_id,
#                     'TeamA_Seed': team_a_seed_num,
#                     'TeamB_Seed': team_b_seed_num,
#                     'SeedDiff': team_a_seed_num - team_b_seed_num,

#                     # Regular season basic stat differentials


#                     # Regular season advanced stat differentials


#                     # Conference tournament stat differentials


#                     # Conference feature differentials


#                     # Coach feature differentials


#                     # Ranking/Massey feature differentials

#                 }

                
#                 matchups.append(features)

#             except Exception as e:
#                 self.logger.warning(f"Error processing game in season {game['Season']}: {str(e)}")
#                 continue

#             return pd.DataFrame(matchups)
        
#     def validate_data(self, data: Dict[str, pd.DataFrame]) -> None:
#         """Validate loaded data for completeness and correctness."""

    
#     def run_pipeline(self, start_year: int = 2003) -> pd.DataFrame:
#         """Execute the full pipeline to create the training dataset."""

#         # # Process each season
#         # for season in sorted(data['tourney']['Season'].unique()):
#         #     self.logger.info(f"Processing season {season}")
            
#         #     # Free memory after each season
#         #     if 'regular_season_stats' in locals():
#         #         del regular_season_stats
            
#         #     # Calculate regular season stats
#         #     regular_season_stats = self.calculate_regular_season_stats(
#         #         data['regular_season'], 
#         #         season
#         #     )
            
#         #     # Create tournament matchups using regular season stats
#         #     tourney_matchups = self.create_tournament_matchups(
#         #         data['tourney'][data['tourney']['Season'] == season],
#         #         regular_season_stats,
#         #         data['seeds']
#         #     )
            
#         #     all_matchups.append(tourney_matchups)
        
#         # final_dataset = pd.concat(all_matchups, ignore_index=True)
        
#         # # Save processed dataset
#         # output_path = self.processed_dir / "tournament_matchups.csv"
#         # final_dataset.to_csv(output_path, index=False)
#         # self.logger.info(f"Saved processed dataset to {output_path}")
        
#         # # Print dataset information
#         # self.logger.info("\nDataset Summary:")
#         # self.logger.info(f"Shape: {final_dataset.shape}")
#         # self.logger.info("\nClass balance:")
#         # self.logger.info(final_dataset['Target'].value_counts(normalize=True))
        
#         # return final_dataset


# if __name__ == "__main__":
#     pipeline = FinalPipeline()
#     dataset = pipeline.run_pipeline()