import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Add the src directory to the Python path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

# Import your pipeline classes
from src.pipelines.cch_feat_pipe import CoachesFeaturePipeline
from src.pipelines.cfc_feat_pipe import ConferencesFeaturePipeline
from src.pipelines.cfc_trny_feat_pipe import ConferenceTourneyFeaturePipeline
from src.pipelines.ranking_feat_pipe import RankingsFeaturePipeline
from src.pipelines.reg_ssn_feat_pipe import RegularSeasonBoxScoreStatsPipeline


class TestPipelines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up logging and create pipeline instances"""
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Initialize pipelines
        cls.coaches_pipeline = CoachesFeaturePipeline()
        cls.conferences_pipeline = ConferencesFeaturePipeline()
        cls.conf_tourney_pipeline = ConferenceTourneyFeaturePipeline()
        cls.rankings_pipeline = RankingsFeaturePipeline()
        cls.regular_season_pipeline = RegularSeasonBoxScoreStatsPipeline()

    def test_coaches_pipeline(self):
        """Test the coaches feature pipeline"""
        self.logger.info("Testing Coaches Pipeline...")
        
        # Run pipeline
        df = self.coaches_pipeline.run_pipeline()
        
        # Basic checks
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        
        # Check required columns
        required_columns = ['Season', 'CoachName', 'TeamID', 'RegularSeasonWinPct', 
                          'TournamentWinPct', 'Experience', 'RegularSeasonGames', 
                          'TournamentGames']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check value ranges
        self.assertTrue(all(0 <= x <= 1 for x in df['RegularSeasonWinPct']))
        self.assertTrue(all(0 <= x <= 1 for x in df['TournamentWinPct']))
        self.assertTrue(all(x >= 0 for x in df['Experience']))
        
        self.logger.info("Coaches Pipeline tests passed")

    def test_conferences_pipeline(self):
        """Test the conferences feature pipeline"""
        self.logger.info("Testing Conferences Pipeline...")
        
        # Run pipeline
        df = self.conferences_pipeline.run_pipeline()
        
        # Basic checks
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        
        # Check required columns
        required_columns = ['Season', 'ConfAbbrev', 'normalized_historical_tourney_points',
                          'historical_tourney_win_pct', 'normalized_ranking_strength']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check value ranges
        self.assertTrue(all(0 <= x <= 1 for x in df['historical_tourney_win_pct']))
        self.assertTrue(all(x >= 0 for x in df['normalized_historical_tourney_points']))
        
        self.logger.info("Conferences Pipeline tests passed")

    def test_conference_tourney_pipeline(self):
        """Test the conference tournament feature pipeline"""
        self.logger.info("Testing Conference Tournament Pipeline...")
        try:
            # Run pipeline
            df = self.conf_tourney_pipeline.run_pipeline()
            
            # Basic checks
            self.assertIsNotNone(df, "Pipeline returned None instead of DataFrame")
            self.assertFalse(df.empty, "Pipeline returned empty DataFrame")
            
            # Check required columns
            required_columns = ['Season', 'TeamID', 'conf_tourney_games', 'conf_tourney_winpct',
                              'conf_tourney_points_ratio', 'days_until_tourney', 
                              'conf_tourney_champion']
            missing_columns = [col for col in required_columns if col not in df.columns]
            self.assertEqual(len(missing_columns), 0, 
                           f"Missing required columns: {missing_columns}")
            
            # Check value ranges
            self.assertTrue(all(0 <= x <= 1 for x in df['conf_tourney_winpct']),
                          "Conference tournament win percentages outside valid range [0,1]")
            self.assertTrue(all(x in [0, 1] for x in df['conf_tourney_champion']),
                          "Conference champion indicator not binary [0,1]")
            self.assertTrue(all(x >= 0 for x in df['conf_tourney_games']),
                          "Negative number of conference tournament games")
            
            # Check for missing values
            self.assertFalse(df[required_columns].isnull().any().any(),
                           "Found missing values in required columns")
            
            self.logger.info("Conference Tournament Pipeline tests passed")
            
        except Exception as e:
            self.logger.error(f"Error in conference tournament pipeline test: {str(e)}")
            raise

    def test_rankings_pipeline(self):
        """Test the rankings feature pipeline"""
        self.logger.info("Testing Rankings Pipeline...")
        
        # Run pipeline
        df = self.rankings_pipeline.run_pipeline()
        
        # Basic checks
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        
        # Check required columns
        required_columns = ['season', 'team_id', 'avg_massey_rank', 'std_massey_rank',
                          'min_massey_rank', 'max_massey_rank', 'median_massey_rank']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check value ranges
        self.assertTrue(all(x > 0 for x in df['avg_massey_rank']))
        self.assertTrue(all(x >= 0 for x in df['std_massey_rank']))
        
        self.logger.info("Rankings Pipeline tests passed")

    def test_regular_season_pipeline(self):
        """Test the regular season stats pipeline"""
        self.logger.info("Testing Regular Season Pipeline...")
        
        # Run pipeline
        df = self.regular_season_pipeline.run_pipeline()
        
        # Basic checks
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        
        # Check required columns
        required_columns = ['season', 'team_id', 'Games', 'Wins', 'WinPct', 
                          'PointsScored_mean', 'PointsAllowed_mean', 'FGPct', 
                          'FG3Pct', 'FTPct', 'OffRating', 'DefRating']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check value ranges
        self.assertTrue(all(0 <= x <= 1 for x in df['WinPct']))
        self.assertTrue(all(0 <= x <= 1 for x in df['FGPct']))
        self.assertTrue(all(x >= 0 for x in df['Games']))
        
        self.logger.info("Regular Season Pipeline tests passed")

    def test_data_consistency(self):
        """Test consistency across all pipeline outputs"""
        self.logger.info("Testing data consistency across pipelines...")
        
        # Run all pipelines
        coaches_df = self.coaches_pipeline.run_pipeline()
        conferences_df = self.conferences_pipeline.run_pipeline()
        conf_tourney_df = self.conf_tourney_pipeline.run_pipeline()
        rankings_df = self.rankings_pipeline.run_pipeline()
        regular_season_df = self.regular_season_pipeline.run_pipeline()
        
        # Check season ranges match
        seasons = {
            'coaches': set(coaches_df['Season'].unique()),
            'conferences': set(conferences_df['Season'].unique()),
            'conf_tourney': set(conf_tourney_df['Season'].unique()),
            'rankings': set(rankings_df['season'].unique()),
            'regular_season': set(regular_season_df['season'].unique())
        }
        
        # All datasets should cover the same seasons
        self.assertEqual(len(set(map(len, seasons.values()))), 1, 
                        "Different number of seasons across datasets")
        
        # Check for missing seasons
        all_seasons = set(range(2003, 2025))
        for name, s in seasons.items():
            missing = all_seasons - s
            if missing:
                self.logger.warning(f"Missing seasons in {name}: {missing}")
        
        self.logger.info("Data consistency tests passed")

if __name__ == '__main__':
    unittest.main(verbosity=2)