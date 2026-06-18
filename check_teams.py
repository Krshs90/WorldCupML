import pandas as pd
from scraper import get_upcoming_fixtures, scrape_historical_data

print("Fetching fixtures...")
fixtures = get_upcoming_fixtures()
espn_teams = set()
for f in fixtures:
    espn_teams.add(f['homeTeam']['name'])
    espn_teams.add(f['awayTeam']['name'])

print(f"Found {len(espn_teams)} teams in fixtures.")

df = scrape_historical_data()
if not df.empty:
    hist_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    print(f"Found {len(hist_teams)} teams in historical data.")
    
    missing = espn_teams - hist_teams
    print(f"Teams in ESPN but missing from historical data: {missing}")
else:
    print("Historical data empty.")
