import sqlite3
import pandas as pd
import sys
import os
sys.path.insert(0, os.getcwd())

from UnitMatchPy.DeepUnitMatch.utils.helpers import pick
from UnitMatchPy.DeepUnitMatch.testing.fast_testing import get_matches_1model


matchtables_db_path = "/Users/suyash/Projects/DeepUnitMatch/matchtables.db"
session_to_find_matches_between = (5, 6)  # RecSes numbers of the two sessions to find matches between

conn = sqlite3.connect(matchtables_db_path)
mt = pd.read_sql_query("SELECT ID1,ID2,RecSes1,RecSes2,NBProb18mice,newISI FROM AL032_19011111882_2", conn)
df = pick(mt, session_to_find_matches_between[0], session_to_find_matches_between[1])
match_indices = get_matches_1model(df, "NBProb18mice")
matches = mt.iloc[match_indices]

print(f'Found {len(matches)} matches between sessions {session_to_find_matches_between[0]} and {session_to_find_matches_between[1]}')


