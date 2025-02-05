import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

class ConferenceTourneyFeaturePipeline:

    def __init__(self, data_dir: str = "data"):
        """Initialize the Conference Tourney feature engineering pipeline with directory structure."""
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

    
    def safe_ratio(self, a, b):
        """
        Safely performs division , taking care of cases where we divide by zero 
        """
        if b == 0:
            return 0
        return a / b

    def extract_conference_tourney_games(self) -> pd.DataFrame:
        """
        Extract all conference tournament games from the regular season detailed results
        and save them as a separate CSV file.
        """
        try:
            self.logger.info("Extracting conference tournament games...")
            
            # Load necessary files
            regular_season = pd.read_csv(self.raw_dir / "MRegularSeasonDetailedResults.csv")
            conf_tourney = pd.read_csv(self.raw_dir / "MConferenceTourneyGames.csv")
            
            # Create unique game identifiers
            conf_tourney['game_id'] = conf_tourney.apply(
                lambda row: f"{row['Season']}_{row['DayNum']}_{row['WTeamID']}_{row['LTeamID']}", 
                axis=1
            )
            
            regular_season['game_id'] = regular_season.apply(
                lambda row: f"{row['Season']}_{row['DayNum']}_{row['WTeamID']}_{row['LTeamID']}", 
                axis=1
            )
            
            # Filter regular season data to get only conference tournament games
            conf_tourney_details = regular_season[
                regular_season['game_id'].isin(conf_tourney['game_id'])
            ].copy()
            
            # Add conference information from conf_tourney DataFrame
            conf_info = conf_tourney[['game_id', 'ConfAbbrev']]
            conf_tourney_details = conf_tourney_details.merge(
                conf_info, 
                on='game_id',
                how='left'
            )
            
            # Validate the extraction
            expected_games = len(conf_tourney)
            actual_games = len(conf_tourney_details)
            
            if expected_games != actual_games:
                self.logger.warning(
                    f"Found {actual_games} games but expected {expected_games} games. "
                    "Some games might be missing."
                )
            
            return conf_tourney_details

            # # Save to CSV
            # output_path = self.raw_dir / "MConferenceTourneyDetailedResults.csv"
            # conf_tourney_details.to_csv(output_path, index=False)
            
            # self.logger.info(f"Saved conference tournament details to {output_path}")
            # self.logger.info(f"Total conference tournament games extracted: {len(conf_tourney_details)}")
            
            # # Print some summary statistics
            # self.logger.info("\nGames by Conference:")
            # conf_summary = conf_tourney_details.groupby('ConfAbbrev').size().sort_values(ascending=False)
            # for conf, count in conf_summary.items():
            #     self.logger.info(f"{conf}: {count} games")
            
            # self.logger.info("\nGames by Season:")
            # season_summary = conf_tourney_details.groupby('Season').size()
            # for season, count in season_summary.items():
            #     self.logger.info(f"{season}: {count} games")
            
            # return conf_tourney_details
            
        except Exception as e:
            self.logger.error(f"Error extracting conference tournament games: {str(e)}")
            return pd.DataFrame()
        
    def calculate_team_features(self, team_games_won: pd.DataFrame, 
                              team_games_lost: pd.DataFrame) -> Dict:
        """Calculate conference tournament features for a single team."""
        
        # If no games played, return None to indicate no tournament participation
        total_games = len(team_games_won) + len(team_games_lost)
        if total_games == 0:
            return None
            
        # Calculate basic stats
        features = {
            'conf_tourney_games': total_games,
            'conf_tourney_winpct': len(team_games_won) / total_games if total_games > 0 else 0
        }
        
        # Calculate points ratio
        points_scored = (
            team_games_won['WScore'].sum() +  # Points when winning
            team_games_lost['LScore'].sum()   # Points when losing
        )
        points_allowed = (
            team_games_won['LScore'].sum() +  # Points allowed when winning
            team_games_lost['WScore'].sum()   # Points allowed when losing
        )
        
        features['conf_tourney_points_ratio'] = (
            points_scored / points_allowed if points_allowed > 0 else 0
        )
        
        # Calculate days rest (using last game's DayNum)
        all_games = pd.concat([
            team_games_won[['DayNum']],
            team_games_lost[['DayNum']]
        ])
        
        if not all_games.empty:
            last_game_day = all_games['DayNum'].max()
            features['days_until_tourney'] = 132 - last_game_day
            
        return features

    def identify_tourney_winners(self, season_games: pd.DataFrame) -> set:
        """Identify teams that won their conference tournament."""
        # Group games by conference
        conf_groups = season_games.groupby('ConfAbbrev')
        winners = set()
        
        for _, conf_games in conf_groups:
            # Find the last day of games for this conference
            last_day = conf_games['DayNum'].max()
            
            # Get games played on the last day
            last_games = conf_games[conf_games['DayNum'] == last_day]
            
            # Winners of the last games are tournament champions
            winners.update(last_games['WTeamID'].unique())
            
        return winners

    def create_features(self, conf_tourney_df: pd.DataFrame) -> pd.DataFrame:
        """Create conference tournament features for all teams across all seasons."""
        
        # Initialize dictionary to store features
        all_features = {}
        
        # Process each season separately
        for season in conf_tourney_df['Season'].unique():
            season_games = conf_tourney_df[conf_tourney_df['Season'] == season]
            
            # Get all teams in this season
            teams = pd.concat([
                season_games['WTeamID'],
                season_games['LTeamID']
            ]).unique()
            
            # Identify conference tournament winners for this season
            tourney_winners = self.identify_tourney_winners(season_games)
            
            # Calculate features for each team
            for team_id in teams:
                # Get all games for this team
                team_games_won = season_games[season_games['WTeamID'] == team_id]
                team_games_lost = season_games[season_games['LTeamID'] == team_id]
                
                # Calculate stats for this team
                team_stats = self.calculate_team_features(team_games_won, team_games_lost)
                
                if team_stats:
                    # Add conference champion flag
                    team_stats['conf_tourney_champion'] = int(team_id in tourney_winners)
                    all_features[(season, team_id)] = team_stats
        
        # Convert to DataFrame
        features_df = pd.DataFrame.from_dict(all_features, orient='index')
        features_df.index = pd.MultiIndex.from_tuples(
            features_df.index, 
            names=['Season', 'TeamID']
        )
        
        return features_df
        
    def run_pipeline(self) -> pd.DataFrame:
        # """Execute the full pipeline to create the team features dataset."""

        # conf_tourney_df = self.extract_conference_tourney_games()
        # conf_tourney_features = self.create_features(conf_tourney_df)

        # # Save features
        # conf_tourney_features.to_csv("data/features/conference_tourney_features.csv")
        
        # # Print sample of features
        # print("\nFeature DataFrame Preview:")
        # print(conf_tourney_features.head())
        # print("\nFeature Statistics:")
        # print(conf_tourney_features.describe())
        """Execute the full pipeline to create the team features dataset."""
        try:
            # Extract conference tournament games
            conf_tourney_df = self.extract_conference_tourney_games()
            if conf_tourney_df.empty:
                self.logger.error("No conference tournament games extracted")
                return pd.DataFrame()

            # Create features
            conf_tourney_features = self.create_features(conf_tourney_df)
            
            # Reset index to make Season and TeamID regular columns
            conf_tourney_features = conf_tourney_features.reset_index()
            
            # Save features
            output_path = self.features_dir / "conference_tourney_features.csv"
            conf_tourney_features.to_csv(output_path, index=False)
            self.logger.info(f"Saved conference tournament features to {output_path}")
            
            # Print sample of features
            self.logger.info("\nFeature DataFrame Preview:")
            self.logger.info(conf_tourney_features.head())
            self.logger.info("\nFeature Statistics:")
            self.logger.info(conf_tourney_features.describe())
            
            return conf_tourney_features
            
        except Exception as e:
            self.logger.error(f"Error in run_pipeline: {str(e)}")
            raise  # Re-raise the exception instead of returning None
        

if __name__ == "__main__":
    pipeline = ConferenceTourneyFeaturePipeline()
    dataset = pipeline.run_pipeline()


        