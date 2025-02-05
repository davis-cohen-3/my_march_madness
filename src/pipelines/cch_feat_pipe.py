import pandas as pd
import numpy as np
import logging
from typing import Dict
from pathlib import Path

class CoachesFeaturePipeline:

    def __init__(self, data_dir: str = "data"):
        """Initialize the Enhanced feature engineering pipeline with directory structure."""
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
            'conferences': pd.read_csv(self.raw_dir / "MTeamConferences.csv"),
            'coaches': pd.read_csv(self.raw_dir / "MTeamCoaches.csv"),
            'massey': pd.read_csv(self.raw_dir / "MMasseyOrdinals_thruSeason2024_day128.csv"),
            'regular_season': pd.read_csv(self.raw_dir / "MRegularSeasonDetailedResults.csv"),
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
    
    def calculate_coach_statistics(self, start_year: int = 2003) -> pd.DataFrame:
        """
        Calculate historical coach statistics from 2003 onwards including regular season win percentage,
        tournament win percentage, and experience. Statistics for each season are calculated
        using only information available up to that point in time.
        """
        try:
            self.logger.info(f"Calculating coach statistics from {start_year} onwards...")
            
            # Load necessary data and filter for seasons >= 2003
            regular_season = pd.read_csv(self.raw_dir / "MRegularSeasonDetailedResults.csv")
            regular_season = regular_season[regular_season['Season'] >= start_year]
            
            tournament = pd.read_csv(self.raw_dir / "MNCAATourneyDetailedResults.csv")
            tournament = tournament[tournament['Season'] >= start_year]
            
            coaches = pd.read_csv(self.raw_dir / "MTeamCoaches.csv")
            coaches = coaches[coaches['Season'] >= start_year]
            
            # Filter for regular season coaches (full season)
            season_coaches = coaches[
                (coaches['FirstDayNum'] == 0) & 
                (coaches['LastDayNum'] == 154)
            ]
            
            # Get unique coach-season pairs
            unique_pairs = season_coaches[['Season', 'CoachName', 'TeamID']].drop_duplicates()
            
            self.logger.info(f"Processing {len(unique_pairs)} coach-season pairs...")
            
            coach_stats = []
            
            for _, row in unique_pairs.iterrows():
                current_season = row['Season']
                coach_name = row['CoachName']
                
                # Get all teams this coach has worked with up through current season
                coach_history = season_coaches[
                    (season_coaches['CoachName'] == coach_name) & 
                    (season_coaches['Season'] <= current_season)
                ]
                coach_teams = coach_history['TeamID'].unique()
                
                # Calculate experience (seasons up through current)
                experience = len(coach_history['Season'].unique())
                
                # Calculate regular season win percentage (including current season)
                reg_season_wins = 0
                reg_season_games = 0
                
                for team_id in coach_teams:
                    team_seasons = coach_history[coach_history['TeamID'] == team_id]['Season'].unique()
                    for season in team_seasons:
                        if season <= current_season:
                            # Wins as the winning team
                            wins = len(regular_season[
                                (regular_season['Season'] == season) & 
                                (regular_season['WTeamID'] == team_id)
                            ])
                            # Losses as the losing team
                            losses = len(regular_season[
                                (regular_season['Season'] == season) & 
                                (regular_season['LTeamID'] == team_id)
                            ])
                            reg_season_wins += wins
                            reg_season_games += wins + losses
                
                reg_season_win_pct = reg_season_wins / reg_season_games if reg_season_games > 0 else 0
                
                # Calculate tournament win percentage (up through previous season)
                tourney_wins = 0
                tourney_games = 0
                
                for team_id in coach_teams:
                    team_seasons = coach_history[coach_history['TeamID'] == team_id]['Season'].unique()
                    for season in team_seasons:
                        if season < current_season:  # Only use previous seasons
                            # Tournament wins
                            wins = len(tournament[
                                (tournament['Season'] == season) & 
                                (tournament['WTeamID'] == team_id)
                            ])
                            # Tournament losses
                            losses = len(tournament[
                                (tournament['Season'] == season) & 
                                (tournament['LTeamID'] == team_id)
                            ])
                            tourney_wins += wins
                            tourney_games += wins + losses
                
                tourney_win_pct = tourney_wins / tourney_games if tourney_games > 0 else 0
                
                coach_stats.append({
                    'Season': current_season,
                    'CoachName': coach_name,
                    'TeamID': row['TeamID'],
                    'RegularSeasonWinPct': round(reg_season_win_pct, 3),
                    'TournamentWinPct': round(tourney_win_pct, 3),
                    'Experience': experience,
                    'RegularSeasonGames': reg_season_games,
                    'TournamentGames': tourney_games
                })
                
                if len(coach_stats) % 100 == 0:
                    self.logger.info(f"Processed {len(coach_stats)} coach-season pairs...")
            
            coach_stats_df = pd.DataFrame(coach_stats)
            
            # Sort by season and coach name
            coach_stats_df = coach_stats_df.sort_values(['Season', 'CoachName'])
            
            # Save to CSV
            output_path = self.features_dir / "coach_features.csv"
            coach_stats_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved coach statistics to {output_path}")
            
            # Print summary statistics
            self.logger.info("\nCoach Statistics Summary:")
            self.logger.info(f"Number of unique coaches: {coach_stats_df['CoachName'].nunique()}")
            self.logger.info(f"Number of seasons covered: {coach_stats_df['Season'].nunique()}")
            self.logger.info("\nExperience Distribution:")
            print(coach_stats_df['Experience'].describe())
            self.logger.info("\nRegular Season Win % Distribution:")
            print(coach_stats_df['RegularSeasonWinPct'].describe())
            self.logger.info("\nTournament Win % Distribution:")
            print(coach_stats_df['TournamentWinPct'].describe())
            
            return coach_stats_df
            
        except Exception as e:
            self.logger.error(f"Error calculating coach statistics: {str(e)}")
            return pd.DataFrame()
        
    def run_pipeline(self, start_year: int = 2003, end_year: int = 2024) -> pd.DataFrame:
        """Execute the full pipeline to create the coaches feature dataset."""
        # Load raw data
        data = self.load_raw_data(start_year)
        
        # Create base features DataFrame with team-season pairs
        features_df = self.create_team_season_pairs(start_year, end_year)
        dataset = self.calculate_coach_statistics()

        return dataset

if __name__ == "__main__":
    pipeline = CoachesFeaturePipeline()
    dataset = pipeline.run_pipeline()