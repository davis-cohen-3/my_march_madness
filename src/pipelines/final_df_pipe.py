import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from collections import defaultdict

class FinalPipeline:
    def __init__(self, data_dir: str = "data"):
        """Initialize the Enhanced feature engineering pipeline with directory structure."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw/mens"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
        
        # Store overlapping features information
        self.overlapping_features = defaultdict(list)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _detect_overlapping_features(self, df1: pd.DataFrame, df2: pd.DataFrame, source1: str, source2: str):
        """Detect and store potentially overlapping features between two dataframes."""
        common_cols = set(df1.columns) & set(df2.columns)
        if common_cols:
            # Remove standard identifier columns
            common_cols = common_cols - {'Season', 'TeamID', 'ConfAbbrev', 'CoachName'}
            if common_cols:
                self.overlapping_features[f"{source1}_vs_{source2}"] = list(common_cols)
                self.logger.info(f"Detected overlapping features between {source1} and {source2}: {common_cols}")

    def _log_merge_stats(self, df: pd.DataFrame, stage: str):
        """Log statistics about the DataFrame at various merge stages."""
        self.logger.info(f"\n=== Merge Stage: {stage} ===")
        self.logger.info(f"Number of rows: {len(df)}")
        self.logger.info(f"Number of unique seasons: {df['Season'].nunique()}")
        self.logger.info(f"Number of unique teams: {df['TeamID'].nunique()}")
        self.logger.info(f"Number of features: {len(df.columns)}")
        self.logger.info(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            self.logger.warning(f"Columns with missing values: {missing_cols}")
            self.logger.warning(f"Missing value counts:\n{df[missing_cols].isnull().sum()}")

    def analyze_coach_gaps(self, merged_df: pd.DataFrame) -> None:
        """Analyze patterns in missing coach data."""
        self.logger.info("\n=== Coach Data Analysis ===")
        
        # Analysis by Season
        self.logger.info("\n1. Missing Coach Data by Season:")
        season_analysis = merged_df.groupby('Season').agg({
            'TeamID': 'count',
            'CoachName': lambda x: x.isna().sum()
        }).rename(columns={
            'TeamID': 'total_teams',
            'CoachName': 'missing_coach_data'
        })
        season_analysis['missing_percentage'] = (
            season_analysis['missing_coach_data'] / season_analysis['total_teams'] * 100
        ).round(2)
        
        for season, row in season_analysis.iterrows():
            if row['missing_coach_data'] > 0:
                self.logger.info(
                    f"Season {season}: {row['missing_coach_data']} missing out of {row['total_teams']} "
                    f"teams ({row['missing_percentage']}%)"
                )
        
        # Analysis by Conference
        self.logger.info("\n2. Missing Coach Data by Conference:")
        conf_analysis = merged_df.groupby('ConfAbbrev').agg({
            'TeamID': 'count',
            'CoachName': lambda x: x.isna().sum()
        }).rename(columns={
            'TeamID': 'total_teams',
            'CoachName': 'missing_coach_data'
        })
        conf_analysis['missing_percentage'] = (
            conf_analysis['missing_coach_data'] / conf_analysis['total_teams'] * 100
        ).round(2)
        
        # Sort by missing percentage
        conf_analysis = conf_analysis.sort_values('missing_percentage', ascending=False)
        
        for conf, row in conf_analysis.iterrows():
            if row['missing_percentage'] > 0:  # Only show conferences with missing data
                self.logger.info(
                    f"Conference {conf}: {row['missing_coach_data']} missing out of {row['total_teams']} "
                    f"teams ({row['missing_percentage']}%)"
                )
        
        # Identify teams with missing coach data
        self.logger.info("\n3. Teams with Missing Coach Data:")
        teams_missing_coach = merged_df[merged_df['CoachName'].isna()].groupby('TeamID').agg({
            'Season': 'count',
            'CoachName': lambda x: x.isna().sum()
        })
        
        if not teams_missing_coach.empty:
            team_names = pd.read_csv(self.raw_dir / "MTeams.csv")
            teams_missing_coach = teams_missing_coach.merge(
                team_names[['TeamID', 'TeamName']], 
                left_index=True, 
                right_on='TeamID'
            )
            
            self.logger.info(f"Found {len(teams_missing_coach)} teams with missing coach data:")
            for _, row in teams_missing_coach.iterrows():
                self.logger.info(
                    f"{row['TeamName']}: Missing {row['Season']} seasons"
                )
                
        # Additional coach-related stats
        self.logger.info("\n4. Additional Coach Statistics:")
        coach_tenure = merged_df[merged_df['Experience'].notna()]['Experience'].describe()
        self.logger.info("\nCoach Experience Distribution:")
        self.logger.info(f"Mean: {coach_tenure['mean']:.2f} seasons")
        self.logger.info(f"Median: {coach_tenure['50%']:.2f} seasons")
        self.logger.info(f"Max: {coach_tenure['max']:.2f} seasons")
        
        win_pct = merged_df[merged_df['RegularSeasonWinPct'].notna()]['RegularSeasonWinPct'].describe()
        self.logger.info("\nRegular Season Win Percentage Distribution:")
        self.logger.info(f"Mean: {win_pct['mean']:.3f}")
        self.logger.info(f"Median: {win_pct['50%']:.3f}")
        self.logger.info(f"StdDev: {win_pct['std']:.3f}")

    def analyze_conference_tourney_gaps(self, merged_df: pd.DataFrame) -> None:
        """Analyze patterns in missing conference tournament data."""
        self.logger.info("\n=== Conference Tournament Data Analysis ===")
        
        # Get set of all team-seasons and those in conference tournaments
        all_team_seasons = set(zip(merged_df['Season'], merged_df['TeamID']))
        tourney_teams = set(zip(merged_df[merged_df['conf_tourney_games'].notna()]['Season'],
                              merged_df[merged_df['conf_tourney_games'].notna()]['TeamID']))
        missing_team_seasons = all_team_seasons - tourney_teams
        
        # Analysis by Season
        self.logger.info("\n1. Missing Data by Season:")
        season_analysis = merged_df.groupby('Season').agg({
            'TeamID': 'count',
            'conf_tourney_games': lambda x: x.isna().sum()
        }).rename(columns={
            'TeamID': 'total_teams',
            'conf_tourney_games': 'missing_tourney_data'
        })
        season_analysis['missing_percentage'] = (
            season_analysis['missing_tourney_data'] / season_analysis['total_teams'] * 100
        ).round(2)
        
        for season, row in season_analysis.iterrows():
            self.logger.info(
                f"Season {season}: {row['missing_tourney_data']} missing out of {row['total_teams']} "
                f"teams ({row['missing_percentage']}%)"
            )
        
        # Analysis by Conference
        self.logger.info("\n2. Missing Data by Conference:")
        conf_analysis = merged_df.groupby('ConfAbbrev').agg({
            'TeamID': 'count',
            'conf_tourney_games': lambda x: x.isna().sum()
        }).rename(columns={
            'TeamID': 'total_teams',
            'conf_tourney_games': 'missing_tourney_data'
        })
        conf_analysis['missing_percentage'] = (
            conf_analysis['missing_tourney_data'] / conf_analysis['total_teams'] * 100
        ).round(2)
        
        # Sort by missing percentage
        conf_analysis = conf_analysis.sort_values('missing_percentage', ascending=False)
        
        for conf, row in conf_analysis.iterrows():
            if row['missing_percentage'] > 0:  # Only show conferences with missing data
                self.logger.info(
                    f"Conference {conf}: {row['missing_tourney_data']} missing out of {row['total_teams']} "
                    f"teams ({row['missing_percentage']}%)"
                )
        
        # Identify teams that never appear in conference tournaments
        self.logger.info("\n3. Teams Never in Conference Tournaments:")
        never_in_tourney = merged_df.groupby('TeamID').agg({
            'Season': 'count',
            'conf_tourney_games': lambda x: x.isna().sum()
        })
        always_missing = never_in_tourney[
            never_in_tourney['Season'] == never_in_tourney['conf_tourney_games']
        ]
        
        if not always_missing.empty:
            team_names = pd.read_csv(self.raw_dir / "MTeams.csv")
            always_missing = always_missing.merge(
                team_names[['TeamID', 'TeamName']], 
                left_index=True, 
                right_on='TeamID'
            )
            
            self.logger.info(f"Found {len(always_missing)} teams never in conference tournaments:")
            for _, row in always_missing.iterrows():
                self.logger.info(
                    f"{row['TeamName']}: Missing all {row['Season']} seasons"
                )
        
        # Special Cases Analysis
        self.logger.info("\n4. Special Cases Analysis:")
        
        # Ivy League Analysis
        ivy_teams = merged_df[
            (merged_df['ConfAbbrev'] == 'ivy') & 
            (merged_df['Season'] < 2017)
        ]
        if not ivy_teams.empty:
            self.logger.info(f"Ivy League pre-2017: {len(ivy_teams)} team-seasons")
            ivy_missing = ivy_teams['conf_tourney_games'].isna().sum()
            self.logger.info(f"Missing conference tournament data: {ivy_missing}")
        
        # Independent Teams Analysis
        independent_teams = merged_df[merged_df['ConfAbbrev'].isna()]
        if not independent_teams.empty:
            self.logger.info(f"\nIndependent Teams Analysis:")
            self.logger.info(f"Total independent team-seasons: {len(independent_teams)}")
            seasons_with_independents = independent_teams.groupby('Season').size()
            self.logger.info("Seasons with independent teams:")
            for season, count in seasons_with_independents.items():
                self.logger.info(f"Season {season}: {count} independent teams")

    def merge_all_features(self) -> pd.DataFrame:
        """Merge all features to store a dataframe representing each team from each season's individual features"""
        self.logger.info("Starting feature merge process...")
        
        try:
            # Load feature datasets
            reg_season_stats = pd.read_csv(self.features_dir / "regular_season_stats.csv")
            rankings = pd.read_csv(self.features_dir / "ranking_features.csv")
            conf_tourney = pd.read_csv(self.features_dir / "conference_tourney_features.csv")
            coach_features = pd.read_csv(self.features_dir / "coach_features.csv")
            conf_features = pd.read_csv(self.features_dir / "conference_features.csv")
            
            # Load team conferences mapping
            team_conferences = pd.read_csv(self.raw_dir / "MTeamConferences.csv")
            
            # Standardize column names
            reg_season_stats = reg_season_stats.rename(columns={
                'team_id': 'TeamID',
                'season': 'Season'
            })
            rankings = rankings.rename(columns={
                'team_id': 'TeamID',
                'season': 'Season'
            })
            
            # Start with regular season stats as base
            self.logger.info("Starting with Regular Season Stats as base")
            merged_df = reg_season_stats.copy()
            self._log_merge_stats(merged_df, "Initial Regular Season Stats")
            
            # First, merge with team conferences to get ConfAbbrev
            self.logger.info("Merging with Team Conferences")
            merged_df = merged_df.merge(
                team_conferences[['Season', 'TeamID', 'ConfAbbrev']],
                on=['Season', 'TeamID'],
                how='left'
            )
            self._log_merge_stats(merged_df, "After Team Conferences Merge")
            
            # Now merge with conference features
            self.logger.info("Merging with Conference Features")
            merged_df = merged_df.merge(
                conf_features,
                on=['Season', 'ConfAbbrev'],
                how='left'
            )
            self._log_merge_stats(merged_df, "After Conference Features Merge")
            
            # Merge with rankings features
            self.logger.info("Merging with Rankings Features")
            self._detect_overlapping_features(merged_df, rankings, "base", "rankings")
            merged_df = merged_df.merge(
                rankings,
                on=['Season', 'TeamID'],
                how='left'
            )
            self._log_merge_stats(merged_df, "After Rankings Merge")
            
            # Merge with conference tournament features
            self.logger.info("Merging with Conference Tournament Features")
            self._detect_overlapping_features(merged_df, conf_tourney, "base", "conf_tourney")
            merged_df = merged_df.merge(
                conf_tourney.reset_index(),
                on=['Season', 'TeamID'],
                how='left'
            )
            self._log_merge_stats(merged_df, "After Conference Tournament Merge")
            
            # Analyze conference tournament gaps
            self.analyze_conference_tourney_gaps(merged_df)
            
            # Merge with coach features
            self.logger.info("Merging with Coach Features")
            self._detect_overlapping_features(merged_df, coach_features, "base", "coaches")
            merged_df = merged_df.merge(
                coach_features,
                on=['Season', 'TeamID'],
                how='left'
            )
            self._log_merge_stats(merged_df, "After Coach Features Merge")
            
            # Analyze coach data gaps
            self.analyze_coach_gaps(merged_df)
            
            # Log overlapping features summary
            self.logger.info("\n=== Overlapping Features Summary ===")
            for comparison, features in self.overlapping_features.items():
                self.logger.info(f"{comparison}: {len(features)} overlapping features")
                self.logger.info(f"Features: {features}\n")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error in merge_all_features: {str(e)}")
            raise

    def run_pipeline(self, start_year: int = 2003) -> pd.DataFrame:
        """Execute the full pipeline to create the merged features dataset."""
        self.logger.info(f"Starting pipeline execution from year {start_year}")
        
        try:
            # Merge all features
            merged_features = self.merge_all_features()
            
            # Filter for desired years
            merged_features = merged_features[merged_features['Season'] >= start_year]
            
            # Save processed dataset
            output_path = self.processed_dir / "team_season_features.csv"
            merged_features.to_csv(output_path, index=False)
            self.logger.info(f"Saved merged features dataset to {output_path}")
            
            # Final dataset summary
            self.logger.info("\n=== Final Dataset Summary ===")
            self.logger.info(f"Total rows: {len(merged_features)}")
            self.logger.info(f"Total features: {len(merged_features.columns)}")
            self.logger.info(f"Years covered: {merged_features['Season'].min()} to {merged_features['Season'].max()}")
            self.logger.info(f"Number of unique teams: {merged_features['TeamID'].nunique()}")
            self.logger.info(f"Number of unique conferences: {merged_features['ConfAbbrev'].nunique()}")
            self.logger.info(f"Number of unique coaches: {merged_features['CoachName'].nunique()}")
            
            return merged_features
            
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = FinalPipeline()
    dataset = pipeline.run_pipeline()