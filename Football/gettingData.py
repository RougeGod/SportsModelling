import nfl_data_py as ndp

bob = ndp.import_pbp_data([2018],["yardline_100", "half_seconds_remaining", "game_half", "down", "goal_to_go", "ydstogo", "posteam", "home_team", "posteam_score", "defteam_score", "play_type"])
bob.to_csv("2018Raw.csv")