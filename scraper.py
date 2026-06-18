import requests
import pandas as pd
import numpy as np

def get_upcoming_fixtures():

    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?limit=100&dates=20260501-20260801"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        matches = []
        if 'events' in data and len(data['events']) > 0:
            for event in data['events']:
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])
                status_info = event.get('status', {}).get('type', {})
                state = status_info.get('state', 'pre')
                
                if len(competitors) == 2:
                    home = competitors[0] if competitors[0]['homeAway'] == 'home' else competitors[1]
                    away = competitors[1] if competitors[0]['homeAway'] == 'home' else competitors[0]
                    
                    matches.append({
                        "id": int(event['id']),
                        "homeTeam": {
                            "name": home['team']['name'], 
                            "logo": home['team'].get('logo', ''),
                            "score": home.get('score', '0')
                        },
                        "awayTeam": {
                            "name": away['team']['name'], 
                            "logo": away['team'].get('logo', ''),
                            "score": away.get('score', '0')
                        },
                        "date": event['date'],
                        "state": state
                    })
            return matches
    except Exception as e:
        print(f"Error fetching fixtures: {e}")
    return []

def scrape_historical_data():
    
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    try:
        print("Downloading raw historical dataset...")
        df = pd.read_csv(url)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        df['is_world_cup'] = (df['tournament'] == 'FIFA World Cup').astype(int)
        df['is_friendly'] = (df['tournament'] == 'Friendly').astype(int)
        df['neutral_venue'] = df['neutral'].astype(int)
        
        print("Engineering advanced Form & H2H features...")

        df['match_id'] = df.index

        df['total_goals'] = df['home_score'] + df['away_score']
        df['goal_diff'] = df['home_score'] - df['away_score']

        df.dropna(subset=['home_score', 'away_score'], inplace=True)
        
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def get_h2h_stats(df, team_a, team_b):
    
    if df is None or df.empty:
        return {}
        
    h2h = df[((df['home_team'] == team_a) & (df['away_team'] == team_b)) | 
             ((df['home_team'] == team_b) & (df['away_team'] == team_a))]
             
    stats = {
        'total_matches': len(h2h),
        'team_a_wins': 0,
        'team_b_wins': 0,
        'draws': 0,
        'avg_goals': 0,
        'recent_meetings': []
    }
    
    if len(h2h) > 0:
        stats['avg_goals'] = round(h2h['total_goals'].mean(), 2)
        
        for _, row in h2h.iterrows():
            if row['home_team'] == team_a:
                if row['home_score'] > row['away_score']: stats['team_a_wins'] += 1
                elif row['home_score'] < row['away_score']: stats['team_b_wins'] += 1
                else: stats['draws'] += 1
            else:
                if row['away_score'] > row['home_score']: stats['team_a_wins'] += 1
                elif row['away_score'] < row['home_score']: stats['team_b_wins'] += 1
                else: stats['draws'] += 1

        recent = h2h.sort_values('date', ascending=False).head(3)
        for _, row in recent.iterrows():
            stats['recent_meetings'].append(f"{row['date'].year}: {row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']}")
            
    return stats

def get_match_details(match_id):
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={match_id}"
    details = {
        'rosters': [],
        'standings': [],
        'predictor': None
    }
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'rosters' in data:
            details['rosters'] = data['rosters']
            
        if 'standings' in data:
            details['standings'] = data['standings']
            
        if 'predictor' in data:
            details['predictor'] = data['predictor']
            
    except Exception as e:
        print(f"Error fetching match details: {e}")
        
    return details
