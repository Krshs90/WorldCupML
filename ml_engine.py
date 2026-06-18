import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

class WorldCupMLEngine:
    def __init__(self):

        self.home_model = None
        self.away_model = None
        self.le_teams = LabelEncoder()
        self.elo_dict = {}
        self.form_dict = {}
        
    def prepare_data(self, df):
        if df.empty: return None, None, None
            
        print("Preparing target variables and advanced features (Elo & Form)...")
        all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
        self.le_teams.fit(all_teams)
        
        df['home_encoded'] = self.le_teams.transform(df['home_team'])
        df['away_encoded'] = self.le_teams.transform(df['away_team'])
        
        elo_dict = {}
        form_dict = {}
        home_elos, away_elos = [], []
        home_forms, away_forms = [], []
        
        for idx, row in df.iterrows():
            h_team, a_team = row['home_team'], row['away_team']
            h_elo, a_elo = elo_dict.get(h_team, 1500), elo_dict.get(a_team, 1500)
            
            home_elos.append(h_elo)
            away_elos.append(a_elo)
            
            h_hist, a_hist = form_dict.get(h_team, []), form_dict.get(a_team, [])
            home_forms.append(sum(h_hist[-5:]) if h_hist else 0)
            away_forms.append(sum(a_hist[-5:]) if a_hist else 0)
            
            h_score, a_score = row['home_score'], row['away_score']
            h_res = 1 if h_score > a_score else (0 if h_score < a_score else 0.5)
            a_res = 1 - h_res
            
            h_exp = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
            a_exp = 1 / (1 + 10 ** ((h_elo - a_elo) / 400))
            
            k = 40 if row.get('is_world_cup', 0) else 20
            
            elo_dict[h_team] = h_elo + k * (h_res - h_exp)
            elo_dict[a_team] = a_elo + k * (a_res - a_exp)
            
            form_dict.setdefault(h_team, []).append(3 if h_res == 1 else (1 if h_res == 0.5 else 0))
            form_dict.setdefault(a_team, []).append(3 if a_res == 1 else (1 if a_res == 0.5 else 0))
            
        df['home_elo'] = home_elos
        df['away_elo'] = away_elos
        df['home_form'] = home_forms
        df['away_form'] = away_forms
        
        self.elo_dict = elo_dict
        self.form_dict = form_dict

        X = df[['home_encoded', 'away_encoded', 'is_world_cup', 'neutral_venue', 'home_elo', 'away_elo', 'home_form', 'away_form']]
        y_home = df['home_score']
        y_away = df['away_score']
        
        return X, y_home, y_away

    def train_model(self, X, y_home, y_away):
        
        X_train, X_test, yh_train, yh_test = train_test_split(X, y_home, test_size=0.1, random_state=42)
        _, _, ya_train, ya_test = train_test_split(X, y_away, test_size=0.1, random_state=42)
        
        self.home_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
        self.away_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
        
        self.home_model.fit(X_train, yh_train)
        self.away_model.fit(X_train, ya_train)

    def predict_match(self, home_team, away_team, iterations=100000, progress_callback=None, live_home_score=0, live_away_score=0):
        
        if self.home_model is None: return None

        team_mapping = {
            "Bosnia-Herzegovina": "Bosnia and Herzegovina",
            "Czechia": "Czech Republic",
            "Congo DR": "DR Congo",
            "Türkiye": "Turkey",
            "United States": "USA",
            "Korea Republic": "South Korea"
        }
        
        mapped_home = team_mapping.get(home_team, home_team)
        mapped_away = team_mapping.get(away_team, away_team)

        try:
            h_enc = self.le_teams.transform([mapped_home])[0]
        except: h_enc = -1
        try:
            a_enc = self.le_teams.transform([mapped_away])[0]
        except: a_enc = -1
            
        if h_enc == -1 or a_enc == -1:
            return {"error": "Cannot simulate: One or both teams are TBD or lack historical data."}
            
        h_elo = self.elo_dict.get(mapped_home, 1500)
        a_elo = self.elo_dict.get(mapped_away, 1500)
        
        h_hist = self.form_dict.get(mapped_home, [])
        a_hist = self.form_dict.get(mapped_away, [])
        h_form = sum(h_hist[-5:]) if h_hist else 0
        a_form = sum(a_hist[-5:]) if a_hist else 0

        input_df = pd.DataFrame({
            'home_encoded': [h_enc], 'away_encoded': [a_enc],
            'is_world_cup': [1], 'neutral_venue': [1],
            'home_elo': [h_elo], 'away_elo': [a_elo],
            'home_form': [h_form], 'away_form': [a_form]
        })
        
        home_xg = max(0.1, self.home_model.predict(input_df)[0])
        away_xg = max(0.1, self.away_model.predict(input_df)[0])

        batch_size = max(10000, iterations // 100)
        completed = 0
        
        home_wins = 0
        draws = 0
        away_wins = 0
        score_counts = {}
        
        while completed < iterations:
            current_batch = min(batch_size, iterations - completed)

            h_sim = np.random.poisson(home_xg, current_batch)
            a_sim = np.random.poisson(away_xg, current_batch)

            h_sim += live_home_score
            a_sim += live_away_score
            
            home_wins += np.sum(h_sim > a_sim)
            draws += np.sum(h_sim == a_sim)
            away_wins += np.sum(h_sim < a_sim)

            combined = h_sim * 1000 + a_sim
            uniques, counts = np.unique(combined, return_counts=True)
            for u, c in zip(uniques, counts):
                h = u // 1000
                a = u % 1000
                score = f"{h}-{a}"
                score_counts[score] = score_counts.get(score, 0) + c
                
            completed += current_batch
            if progress_callback: progress_callback(completed)

        all_scores = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)

        all_scores = all_scores[:200]
        
        all_scores_pct = [(s, round((c/iterations)*100, 3)) for s, c in all_scores]
        
        return {
            "home_win": round((home_wins / iterations) * 100, 1),
            "draw": round((draws / iterations) * 100, 1),
            "away_win": round((away_wins / iterations) * 100, 1),
            "scorelines": all_scores_pct,
            "home_xg": round(float(home_xg), 2),
            "away_xg": round(float(away_xg), 2)
        }
