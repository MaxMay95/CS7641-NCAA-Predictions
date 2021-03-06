import os
import pandas as pd
import numpy as np


def get_game_data(relative=None, norm=None, stats_filter=None, features_only=None, datasets_dir=None, games_file_name=None, stats_file_name=None):
    """
    Combine game data and regular season statistics to form a DataFrame where each row represents one game from an NCAA
    tournament. Ensure the Team1 and Team2 values are randomly swapped in games_file_name so that the first team is not
    always the higher seed. (Done already for provided data).

    :param relative: If True, use relative change in each category [(Team A - Team B)/avg(Team A, Team B)]. If False,
        use absolute change. Defaults to True.
    :param norm: If True, keep only Norm features (that is, features normalized over the tournament teams from each
        year). Overridden by stats_filter. Defaults to True.
    :param stats_filter: A function to filter which statistics are kept. Takes a list of stat names and should return
        the filtered list. If None, keep all stats. Defaults to None.
    :param features_only: If True, only keep features and results for use in training. This will discard Team IDs,
        round information, year, and game number. Defaults to True.
    :param datasets_dir: Name of the directory in cwd containing the CSVs with games_file and stats_file. Defaults to
        'datasets'.
    :param games_file_name: Name of the CSV file containing game info (which teams played in each game).
    :param stats_file_name: Name of the CSV file containing regular season statistics for each team.
    :return: pandas DataFrame containing data for each game detailed in games_file_name. Each row represents one game
        from an NCAA tournament, with statistics in the form Team A - Team B (or relative difference if relative=True).
    """
    relative = True if relative is None else relative
    norm = True if norm is None else norm
    features_only = True if features_only is None else features_only
    datasets_dir = 'datasets' if datasets_dir is None else datasets_dir
    games_file_name = 'NCAA tournament games 2010-2018 shuffled.csv' if games_file_name is None else games_file_name
    if games_file_name[-4:] != '.csv':
        raise Warning('games_file_name must end in .csv. A CSV file is required.')
    stats_file_name = 'Stats by team and year 2010-2018.csv' if stats_file_name is None else stats_file_name
    if stats_file_name[-4:] != '.csv':
        raise Warning('stats_file_name must end in .csv. A CSV file is required.')
    games_file_path = os.path.join(datasets_dir, games_file_name)
    stats_file_path = os.path.join(datasets_dir, stats_file_name)

    df_games = pd.read_csv(games_file_path, sep=',')
    df_games = df_games.loc[:, ~df_games.columns.str.contains('^Unnamed')]
    df_stats = pd.read_csv(stats_file_path, sep=',')
    df_stats = df_stats.loc[:, ~df_stats.columns.str.contains('^Unnamed')]
    if stats_filter is not None:
        stats = list(df_stats.columns.drop(['School ID', 'Year']))
        stats = stats_filter(stats)
        stats.insert(0, 'School ID')
        df_stats = df_stats[stats]
    elif norm:
        stats = list(df_stats.columns.drop(['School ID', 'Year']))
        stats = [stat for stat in stats if 'Norm' in stat]
        stats.insert(0, 'School ID')
        df_stats = df_stats[stats]
    else:
        stats = list(df_stats.columns.drop(['School ID', 'Year']))
        stats = [stat for stat in stats if 'Norm' not in stat]
        stats.insert(0, 'School ID')
        df_stats = df_stats[stats]
    df_1 = df_games.join(df_stats.set_index('School ID'), on='Team1 ID')
    df_2 = df_games.join(df_stats.set_index('School ID'), on='Team2 ID')
    df_stats_1 = df_1.loc[:, 'Game Num':].drop('Game Num', axis=1)
    df_stats_2 = df_2.loc[:, 'Game Num':].drop('Game Num', axis=1)
    if relative:
        df_game_data = df_games.join(((df_stats_1 - df_stats_2) / ((df_stats_1 + df_stats_2) / 2)).add_suffix(' Diff'))
    else:
        df_game_data = df_games.join((df_stats_1 - df_stats_2).add_suffix(' Diff'))
    if features_only:
        df_game_data = df_game_data.drop(columns=['Year', 'Team1 ID', 'Team2 ID', 'Game Num', 'Round Num', 'Round'])
    return df_game_data


def shuffle_games(datasets_dir=None, games_file_name=None, games_file_new_name=None, seed=None):
    """
    Randomly swap Team1 and Team2 (and thus the value of Team1 Result) in games_file.

    :param datasets_dir: Name of the directory in cwd containing the CSVs with games_file and stats_file. Defaults to
        'datasets'.
    :param games_file_name: Name of the CSV file containing game info (which teams played in each game).
    :param games_file_new_name: Name of the CSV file to save the shuffled games into. Defaults to games_file_name with
        ' shuffled' appended before .csv.
    :param seed: Random seed.
    """
    datasets_dir = 'datasets' if datasets_dir is None else datasets_dir
    games_file_name = 'NCAA tournament games 2010-2018.csv' if games_file_name is None else games_file_name
    if games_file_name[-4:] != '.csv':
        raise Warning('games_file_name must end in .csv. A CSV file is required.')
    games_file_new_name = games_file_name[:-4] + ' shuffled.csv' if games_file_new_name is None else games_file_new_name
    if games_file_new_name[-4:] != '.csv':
        raise Warning('games_file_new_name must end in .csv. A CSV file is required.')
    games_file_path = os.path.join(datasets_dir, games_file_name)
    games_file_new_path = os.path.join(datasets_dir, games_file_new_name)
    df_games = pd.read_csv(games_file_path, sep=',')
    df_games = df_games.loc[:, ~df_games.columns.str.contains('^Unnamed')]
    if seed is not None:
        np.random.seed(seed)
    for index, row in df_games.iterrows():
        if np.random.choice([True, False]):
            df_games.at[index, 'Team1 ID'], df_games.at[index, 'Team2 ID'] = row['Team2 ID'], row['Team1 ID']
            df_games.at[index, 'Team1 result'] = 'win' if row['Team1 result'] == 'loss' else 'loss'
    df_games.to_csv(games_file_new_path, index=False)
