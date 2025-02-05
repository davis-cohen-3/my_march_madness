import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

class RankingsFeaturePipeline:

    def __init__(self, data_dir: str = "data"):
        """Initialize the Rankings feature engineering pipeline with directory structure."""
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
            'massey': pd.read_csv(self.raw_dir / "MMasseyOrdinals_thruSeason2024_day128.csv")
        }
        
        return data_files
    
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
    
    def add_massey_features(self, features_df: pd.DataFrame, massey_df: pd.DataFrame) -> pd.DataFrame:
        """Add Massey Ordinals ranking features."""
        self.logger.info("Adding Massey ranking features...")
        features = features_df.copy()
        
        # Prepare Massey data
        massey = massey_df.rename(columns={
            'Season': 'season',
            'TeamID': 'team_id'
        })
        
        # Calculate ranking statistics
        ranking_stats = massey.groupby(['season', 'team_id'])['OrdinalRank'].agg([
            'mean',
            'std',
            'min',
            'max',
            'median'
        ]).round(2)
        
        # Rename columns
        ranking_stats = ranking_stats.rename(columns={
            'mean': 'avg_massey_rank',
            'std': 'std_massey_rank',
            'min': 'min_massey_rank',
            'max': 'max_massey_rank',
            'median': 'median_massey_rank'
        }).reset_index()
        
        # Merge with features
        features = features.merge(
            ranking_stats,
            on=['season', 'team_id'],
            how='left'
        )
        
        # Fill missing values
        fill_values = {
            'avg_massey_rank': 400,
            'std_massey_rank': 0,
            'min_massey_rank': 400,
            'max_massey_rank': 400,
            'median_massey_rank': 400
        }
        features = features.fillna(fill_values)
        
        return features
    
    def run_pipeline(self, start_year: int = 2003, end_year: int = 2024) -> pd.DataFrame:
        """Execute the full pipeline to create the team features dataset."""
        # Load raw data
        data = self.load_raw_data(start_year)
        
        # Create base features DataFrame with team-season pairs
        features_df = self.create_team_season_pairs(start_year, end_year)
        
        # Add massey features
        features_df = self.add_massey_features(features_df, data['massey'])

        # Save processed dataset
        output_path = self.features_dir / "ranking_features.csv"
        features_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved Ranking features dataset to {output_path}")
        
        # Print dataset information
        self.logger.info("\nDataset Summary:")
        self.logger.info(f"Shape: {features_df.shape}")
        self.logger.info("\nFeatures included:")
        self.logger.info(f"Columns: {features_df.columns.tolist()}")
        
        return features_df
    
if __name__ == "__main__":
    pipeline = RankingsFeaturePipeline()
    dataset = pipeline.run_pipeline()
