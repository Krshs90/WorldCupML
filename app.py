import customtkinter as ctk
from PIL import Image
import threading
import requests
from io import BytesIO
from datetime import datetime

from scraper import get_upcoming_fixtures, scrape_historical_data, get_h2h_stats, get_match_details
from ml_engine import WorldCupMLEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class WorldCupMLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("WorldCupML - V1 Predictor")
        self.geometry("1100x850")
        
        self.ml_engine = WorldCupMLEngine()
        self.historical_data = None
        self.fixtures = []
        self.selected_match = None
        self.is_training = False
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=350, corner_radius=0, fg_color="#12161B")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(3, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="WorldCupML 🏆\nV1 Engine", font=ctk.CTkFont(size=28, weight="bold", family="Helvetica"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))
        
        self.refresh_btn = ctk.CTkButton(self.sidebar_frame, text="Refresh Fixtures", command=self.load_fixtures_async, fg_color="#1F6AA5", hover_color="#144870")
        self.refresh_btn.grid(row=1, column=0, padx=20, pady=(0, 10))
        
        self.hide_finished_var = ctk.BooleanVar(value=True)
        self.hide_finished_switch = ctk.CTkSwitch(self.sidebar_frame, text="Hide Finished Matches", variable=self.hide_finished_var, command=self.update_sidebar, progress_color="#10B981")
        self.hide_finished_switch.grid(row=2, column=0, padx=20, pady=(0, 5))
        
        self.scrollable_matches = ctk.CTkScrollableFrame(self.sidebar_frame, label_text="Tournament Matches", fg_color="#1E232A", scrollbar_button_color="#2C333D")
        self.scrollable_matches.grid(row=3, column=0, padx=15, pady=(5, 15), sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(3, weight=1)

        self.main_scrollable = ctk.CTkScrollableFrame(self, corner_radius=15, fg_color="#171A21")
        self.main_scrollable.grid(row=0, column=1, padx=25, pady=25, sticky="nsew")
        self.main_scrollable.grid_columnconfigure(0, weight=1)

        self.header_frame = ctk.CTkFrame(self.main_scrollable, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, pady=(30, 10))
        
        self.home_logo_lbl = ctk.CTkLabel(self.header_frame, text="")
        self.home_logo_lbl.grid(row=0, column=0, padx=30)
        
        self.vs_lbl = ctk.CTkLabel(self.header_frame, text="VS", font=ctk.CTkFont(size=36, weight="bold"), text_color="#3B82F6")
        self.vs_lbl.grid(row=0, column=1, padx=30)
        
        self.away_logo_lbl = ctk.CTkLabel(self.header_frame, text="")
        self.away_logo_lbl.grid(row=0, column=2, padx=30)
        
        self.match_title = ctk.CTkLabel(self.main_scrollable, text="Select a Match from the Sidebar", font=ctk.CTkFont(size=32, weight="bold"))
        self.match_title.grid(row=1, column=0, pady=(10, 5))
        
        self.match_date_lbl = ctk.CTkLabel(self.main_scrollable, text="", font=ctk.CTkFont(size=16), text_color="#9CA3AF")
        self.match_date_lbl.grid(row=2, column=0, pady=(0, 10))

        self.tabview = ctk.CTkTabview(self.main_scrollable)
        self.tabview.grid(row=3, column=0, pady=10, padx=20, sticky="nsew")
        self.main_scrollable.grid_rowconfigure(3, weight=1)
        
        self.tab_sim = self.tabview.add("Simulation & Prediction")
        self.tab_lineups = self.tabview.add("Lineups & Squads")
        self.tab_stats = self.tabview.add("Standings & Stats")
        
        self.tab_sim.grid_columnconfigure(0, weight=1)
        self.tab_lineups.grid_columnconfigure(0, weight=1)
        self.tab_stats.grid_columnconfigure(0, weight=1)

        self.h2h_frame = ctk.CTkFrame(self.tab_sim, fg_color="#12161B", corner_radius=10, border_width=1, border_color="#2C333D")
        self.h2h_text = ctk.CTkLabel(self.h2h_frame, text="Waiting for match selection...", font=ctk.CTkFont(size=14), text_color="#A0AEC0", justify="center")
        self.h2h_text.pack(pady=15, padx=25)
        self.h2h_frame.grid(row=0, column=0, pady=10, padx=20, sticky="ew")

        self.training_frame = ctk.CTkFrame(self.tab_sim, fg_color="#1E232A", corner_radius=10)
        self.training_frame.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        self.training_frame.grid_columnconfigure(0, weight=1)
        
        self.sim_label = ctk.CTkLabel(self.training_frame, text="Monte Carlo Iterations (Min: 100,000):", font=ctk.CTkFont(size=15, weight="bold"))
        self.sim_label.grid(row=0, column=0, pady=(15, 5))
        
        self.sim_slider = ctk.CTkSlider(self.training_frame, from_=100000, to=100000000, number_of_steps=999, command=self.update_entry_from_slider, button_color="#3B82F6")
        self.sim_slider.grid(row=1, column=0, sticky="ew", padx=30, pady=(0, 10))
        self.sim_slider.set(1000000)
        
        self.sim_entry = ctk.CTkEntry(self.training_frame, width=200, justify="center", font=ctk.CTkFont(size=14))
        self.sim_entry.insert(0, "1000000")
        self.sim_entry.grid(row=2, column=0, pady=(0, 20))
        
        self.progress_label = ctk.CTkLabel(self.training_frame, text="Scenarios Simulated: 0", font=ctk.CTkFont(size=16))
        self.progress_bar = ctk.CTkProgressBar(self.training_frame, progress_color="#10B981")
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self.training_frame, text="", font=ctk.CTkFont(size=14))
        
        self.btn_frame = ctk.CTkFrame(self.training_frame, fg_color="transparent")
        self.train_btn = ctk.CTkButton(self.btn_frame, text="Run XGBoost Regressor", command=self.start_training_thread, height=45, font=ctk.CTkFont(size=15, weight="bold"))
        self.train_btn.grid(row=0, column=0, padx=15)
        
        self.clear_btn = ctk.CTkButton(self.btn_frame, text="Clear Data", command=self.clear_training, fg_color="#EF4444", hover_color="#B91C1C", height=45, font=ctk.CTkFont(size=15, weight="bold"))
        self.clear_btn.grid(row=0, column=1, padx=15)

        self.results_frame = ctk.CTkFrame(self.tab_sim, fg_color="#12161B", corner_radius=10, border_width=1, border_color="#3B82F6")
        self.results_label = ctk.CTkLabel(self.results_frame, text="Wins and Score Probabilities", font=ctk.CTkFont(size=22, weight="bold"))
        
        self.home_win_lbl = ctk.CTkLabel(self.results_frame, text="")
        self.draw_lbl = ctk.CTkLabel(self.results_frame, text="")
        self.away_win_lbl = ctk.CTkLabel(self.results_frame, text="")
        
        self.scorelines_lbl = ctk.CTkLabel(self.results_frame, text="", font=ctk.CTkFont(size=18))
        
        self.show_all_btn = ctk.CTkButton(self.results_frame, text="Show All Probabilities ▼", command=self.toggle_all_scores, fg_color="transparent", border_width=1, hover_color="#1E232A", text_color="#3B82F6")
        self.all_scores_frame = ctk.CTkScrollableFrame(self.results_frame, height=250, fg_color="#171A21")
        self.all_scores_lbl = ctk.CTkLabel(self.all_scores_frame, text="", font=ctk.CTkFont(size=14), justify="left")
        self.all_scores_lbl.pack(padx=20, pady=10)
        self.is_showing_all = False

        self.lineup_scroll = ctk.CTkScrollableFrame(self.tab_lineups, fg_color="transparent")
        self.lineup_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.standings_scroll = ctk.CTkScrollableFrame(self.tab_stats, fg_color="transparent")
        self.standings_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.load_fixtures_async()
        
    def update_entry_from_slider(self, val):
        self.sim_entry.delete(0, 'end')
        self.sim_entry.insert(0, str(int(val)))
        
    def load_fixtures_async(self):
        self.refresh_btn.configure(state="disabled", text="Loading Matches...")
        for widget in self.scrollable_matches.winfo_children():
            widget.destroy()
            
        def fetch():
            self.fixtures = get_upcoming_fixtures()
            if self.historical_data is None:
                self.historical_data = scrape_historical_data()
            self.after(0, self.update_sidebar)
            
        threading.Thread(target=fetch, daemon=True).start()
        
    def update_sidebar(self):
        self.refresh_btn.configure(state="normal", text="Refresh Fixtures")
        if not self.fixtures:
            ctk.CTkLabel(self.scrollable_matches, text="No matches found.\nFetching more data...").pack(pady=20)
            return
            
        for match in self.fixtures:
            if self.hide_finished_var.get() and match['state'] == 'post':
                continue
                
            btn_text = f"{match['homeTeam']['name']} vs {match['awayTeam']['name']}"
            if match['state'] == 'post':
                btn_text = f"[DONE] {btn_text}"
            elif match['state'] == 'in':
                btn_text = f"🔴 [LIVE] {btn_text}"
                
            btn = ctk.CTkButton(self.scrollable_matches, text=btn_text, fg_color="#2C333D", hover_color="#3A4350",
                                border_width=0, text_color="#DCE4EE", font=ctk.CTkFont(size=14, weight="bold"),
                                height=40, command=lambda m=match: self.select_match_async(m))
            btn.pack(pady=8, padx=8, fill="x")

    def load_image_from_url(self, url, size=(100, 100)):
        if not url: return None
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content))
            return ctk.CTkImage(light_image=img, dark_image=img, size=size)
        except:
            return None

    def async_load_image_to_label(self, url, label, size=(30, 30)):
        if not url: return
        def fetch():
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
                def update_ui():
                    try:
                        if label.winfo_exists():
                            label.configure(image=ctk_img, text="")
                    except:
                        pass
                self.after(0, update_ui)
            except:
                pass
        threading.Thread(target=fetch, daemon=True).start()

    def format_date(self, date_str):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%MZ")
            return dt.strftime("%B %d, %Y at %H:%M UTC")
        except:
            return date_str

    def select_match_async(self, match):
        self.selected_match = match
        self.match_title.configure(text="Loading Match Data...")
        self.clear_training()
        
        def load_details():
            h_name = match['homeTeam']['name']
            a_name = match['awayTeam']['name']
            h_img = self.load_image_from_url(match['homeTeam']['logo'])
            a_img = self.load_image_from_url(match['awayTeam']['logo'])
            stats = get_h2h_stats(self.historical_data, h_name, a_name)
            details = get_match_details(match['id'])
            
            self.after(0, lambda: self.update_match_ui(h_name, a_name, h_img, a_img, stats, match, details))
            
        threading.Thread(target=load_details, daemon=True).start()
        
    def render_lineups(self, rosters):
        for widget in self.lineup_scroll.winfo_children():
            widget.destroy()
            
        if not rosters or len(rosters) == 0:
            ctk.CTkLabel(self.lineup_scroll, text="No lineup/squad data available yet for this match.", text_color="#9CA3AF").pack(pady=40)
            return

        columns_frame = ctk.CTkFrame(self.lineup_scroll, fg_color="transparent")
        columns_frame.pack(fill="both", expand=True)
        columns_frame.grid_columnconfigure((0, 1), weight=1)
        
        for idx, roster in enumerate(rosters[:2]):
            team_name = roster.get('team', {}).get('displayName', 'Unknown Team')
            formation = roster.get('formation', '')
            
            col = ctk.CTkFrame(columns_frame, fg_color="#1E232A", corner_radius=10)
            col.grid(row=0, column=idx, padx=10, pady=10, sticky="nsew")
            
            title_text = f"{team_name}"
            if formation: title_text += f" ({formation})"
            
            ctk.CTkLabel(col, text=title_text, font=ctk.CTkFont(size=18, weight="bold"), text_color="#3B82F6").pack(pady=10)
            
            if 'roster' in roster and roster['roster']:
                starters = [p for p in roster['roster'] if p.get('starter', False)]
                subs = [p for p in roster['roster'] if not p.get('starter', False)]
                
                def render_player_group(title, players):
                    if not players: return
                    ctk.CTkLabel(col, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color="#A0AEC0", anchor="w").pack(fill="x", padx=15, pady=(15, 5))
                    
                    for player in players:
                        p_frame = ctk.CTkFrame(col, fg_color="#171A21", corner_radius=8, height=45)
                        p_frame.pack(fill="x", padx=10, pady=4)
                        p_frame.pack_propagate(False)
                        
                        p_info = player.get('athlete', {})
                        name = p_info.get('displayName', 'Unknown Player')
                        jersey = p_info.get('jersey', '')
                        pos = player.get('position', {}).get('abbreviation', 'Unk')
                        headshot_url = p_info.get('headshot', {}).get('href', '')
                        
                        ctk.CTkLabel(p_frame, text=f"{jersey}", font=ctk.CTkFont(size=14, weight="bold"), text_color="#4B5563", width=25).pack(side="left", padx=(10, 5))
                        
                        img_label = ctk.CTkLabel(p_frame, text="👤", width=30, height=30)
                        img_label.pack(side="left", padx=5)
                        if headshot_url:
                            self.async_load_image_to_label(headshot_url, img_label, size=(30, 30))
                        
                        ctk.CTkLabel(p_frame, text=name, font=ctk.CTkFont(size=14, weight="bold"), text_color="#E5E7EB", anchor="w").pack(side="left", padx=10)
                        ctk.CTkLabel(p_frame, text=pos, font=ctk.CTkFont(size=12, weight="bold"), text_color="#3B82F6", anchor="e").pack(side="right", padx=15)
                        
                render_player_group("Starting XI", starters)
                render_player_group("Substitutes", subs)
            else:
                ctk.CTkLabel(col, text="Roster empty", text_color="#9CA3AF").pack(pady=20)

    def render_standings(self, standings):
        for widget in self.standings_scroll.winfo_children():
            widget.destroy()
            
        if not standings:
            ctk.CTkLabel(self.standings_scroll, text="No standings available for this match.", text_color="#9CA3AF").pack(pady=40)
            return
            
        if isinstance(standings, dict):
            groups_data = standings.get('groups', [standings])
        elif isinstance(standings, list):
            groups_data = standings
        else:
            groups_data = []
            
        if not groups_data or len(groups_data) == 0:
            ctk.CTkLabel(self.standings_scroll, text="No standings available for this match.", text_color="#9CA3AF").pack(pady=40)
            return
            
        for group in groups_data:
            if isinstance(group, str): continue
            
            group_name = group.get('name', 'Group Standings')
            g_frame = ctk.CTkFrame(self.standings_scroll, fg_color="#1E232A", corner_radius=10)
            g_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(g_frame, text=group_name, font=ctk.CTkFont(size=18, weight="bold"), text_color="#10B981").pack(pady=10)
            
            headers = ["Team", "GP", "W", "D", "L", "GF", "GA", "GD", "Pts"]
            h_frame = ctk.CTkFrame(g_frame, fg_color="#171A21", corner_radius=5)
            h_frame.pack(fill="x", padx=10, pady=5)
            for i, h in enumerate(headers):
                width = 250 if i == 0 else 50
                anchor = "w" if i == 0 else "center"
                pad = (20, 0) if i == 0 else 0
                ctk.CTkLabel(h_frame, text=h, font=ctk.CTkFont(size=13, weight="bold"), text_color="#9CA3AF", width=width, anchor=anchor).grid(row=0, column=i, padx=pad, pady=5)
                
            if 'standings' in group:
                s_list = group['standings'] if isinstance(group['standings'], list) else [group['standings']]
                entries = s_list[0].get('entries', []) if len(s_list) > 0 else []
                for r_idx, entry in enumerate(entries):
                    row_frame = ctk.CTkFrame(g_frame, fg_color="transparent", height=40)
                    row_frame.pack(fill="x", padx=10, pady=1)
                    row_frame.pack_propagate(False)
                    
                    team_info = entry.get('team', {}) if isinstance(entry, dict) else {}
                    if isinstance(team_info, str):
                        team_name = team_info
                        logos = []
                    else:
                        team_name = team_info.get('displayName', 'Unknown')
                        logos = team_info.get('logos', [])
                        
                    logo_url = logos[0].get('href', '') if logos else ''
                    
                    team_frame = ctk.CTkFrame(row_frame, fg_color="transparent", width=250)
                    team_frame.grid(row=0, column=0, padx=(20, 0), sticky="w")
                    team_frame.grid_propagate(False)
                    
                    logo_lbl = ctk.CTkLabel(team_frame, text="🛡️", width=25, height=25)
                    logo_lbl.pack(side="left", padx=(0, 10))
                    if logo_url:
                        self.async_load_image_to_label(logo_url, logo_lbl, size=(25, 25))
                        
                    name_lbl = ctk.CTkLabel(team_frame, text=team_name, font=ctk.CTkFont(size=14, weight="bold"), text_color="#E5E7EB", anchor="w")
                    name_lbl.pack(side="left", fill="x")
                    
                    stats = entry.get('stats', [])
                    
                    def get_stat(name):
                        for s in stats:
                            if s.get('name') == name: return s.get('displayValue', '0')
                        return '0'
                        
                    gp = get_stat('gamesPlayed')
                    w = get_stat('wins')
                    d = get_stat('ties')
                    l = get_stat('losses')
                    gf = get_stat('pointsFor')
                    ga = get_stat('pointsAgainst')
                    gd = get_stat('pointDifferential')
                    pts = get_stat('points')
                    
                    vals = [team_name, gp, w, d, l, gf, ga, gd, pts]
                    for i, val in enumerate(vals):
                        if i == 0: continue
                        
                        width = 50
                        color = "#10B981" if i == 8 else "#E5E7EB"
                        weight = "bold" if i == 8 else "normal"
                        ctk.CTkLabel(row_frame, text=str(val), font=ctk.CTkFont(size=14, weight=weight), width=width, anchor="center", text_color=color).grid(row=0, column=i)

    def update_match_ui(self, h_name, a_name, h_img, a_img, stats, match, details):
        self.match_title.configure(text=f"{h_name} vs {a_name}")
        self.match_date_lbl.configure(text=f"Match Date: {self.format_date(match['date'])}")

        h_score = match['homeTeam'].get('score', '0')
        a_score = match['awayTeam'].get('score', '0')
        
        if match['state'] == 'post':
            self.vs_lbl.configure(text=f"{h_score} - {a_score}\n(FINISHED)", text_color="#EF4444")
            self.train_btn.configure(state="disabled", text="Match Already Finished")
            self.status_label.configure(text="Cannot simulate a match that has already ended.", text_color="#EF4444")
        elif match['state'] == 'in':
            self.vs_lbl.configure(text=f"{h_score} - {a_score}\n🔴 LIVE", text_color="#10B981")
            self.train_btn.configure(state="normal", text="Run XGBoost Regressor (Live)")
            self.status_label.configure(text="Simulations will factor in the current live score.", text_color="#10B981")
        else:
            self.vs_lbl.configure(text="VS", text_color="#3B82F6")
            self.train_btn.configure(state="normal", text="Run XGBoost Regressor")
            self.status_label.configure(text="", text_color="white")
        
        if h_img:
            self.home_logo_lbl.configure(image=h_img, text="")
        else:
            self.home_logo_lbl.configure(image=None, text=h_name, font=ctk.CTkFont(size=24, weight="bold"))
            
        if a_img:
            self.away_logo_lbl.configure(image=a_img, text="")
        else:
            self.away_logo_lbl.configure(image=None, text=a_name, font=ctk.CTkFont(size=24, weight="bold"))
            
        if stats and stats.get('total_matches', 0) > 0:
            h2h_str = f"📚 Historical H2H Context (World Cup / Friendlies)\n\n"
            h2h_str += f"Total Encounters: {stats['total_matches']}  |  {h_name} Wins: {stats['team_a_wins']}  |  {a_name} Wins: {stats['team_b_wins']}  |  Draws: {stats['draws']}\n\n"
            if stats['recent_meetings']:
                h2h_str += "Recent Matchups:   " + "  —  ".join(stats['recent_meetings'])
        else:
            h2h_str = "No historical matchup data found in our dataset."
            
        self.h2h_text.configure(text=h2h_str)

        if details:
            self.render_lineups(details.get('rosters'))
            self.render_standings(details.get('standings'))
        else:
            self.render_lineups([])
            self.render_standings([])
        
        self.progress_label.grid(row=3, column=0, pady=(0, 10))
        self.progress_bar.grid(row=4, column=0, sticky="ew", padx=30, pady=(0, 10))
        self.status_label.grid(row=5, column=0, pady=(0, 20))
        self.btn_frame.grid(row=6, column=0, pady=(0, 20))
        self.results_frame.grid_forget()

    def update_progress(self, current):
        try:
            target = int(self.sim_entry.get().replace(',', ''))
        except:
            target = 100000
        self.after(0, lambda: self.progress_bar.set(current / target))
        self.after(0, lambda: self.progress_label.configure(text=f"Scenarios Simulated: {current:,}"))
        if current >= target:
            self.after(0, lambda: self.status_label.configure(text="Monte Carlo Target Reached ✅", text_color="#10B981", font=ctk.CTkFont(weight="bold")))

    def start_training_thread(self):
        if not self.selected_match or self.is_training or self.selected_match['state'] == 'post':
            return
            
        self.is_training = True
        self.train_btn.configure(state="disabled", text="Simulating...")
        self.status_label.configure(text="Training Regressors & Generating Poisson Distribution...", text_color="#FBBF24")
        
        def run_ml():
            try:
                raw_val = self.sim_entry.get().replace(',', '').replace(' ', '')
                iterations = int(raw_val)
                if iterations < 100000:
                    iterations = 100000
                    self.after(0, lambda: self.sim_entry.delete(0, 'end'))
                    self.after(0, lambda: self.sim_entry.insert(0, "100000"))
            except ValueError:
                self.after(0, lambda: self.status_label.configure(text="Error: Please enter a valid number.", text_color="#EF4444"))
                self.is_training = False
                self.after(0, lambda: self.train_btn.configure(state="normal", text="Run XGBoost Regressor"))
                return
                
            X, y_home, y_away = self.ml_engine.prepare_data(self.historical_data)
            
            self.after(0, lambda: self.status_label.configure(text="Training XGBoost Model (Analyzing Data)...", text_color="#FBBF24"))
            self.ml_engine.train_model(X, y_home, y_away)
            
            self.after(0, lambda: self.status_label.configure(text="Running Monte Carlo Simulations...", text_color="#FBBF24"))
            h_name = self.selected_match['homeTeam']['name']
            a_name = self.selected_match['awayTeam']['name']
            
            live_h = int(self.selected_match['homeTeam'].get('score', 0)) if self.selected_match['state'] == 'in' else 0
            live_a = int(self.selected_match['awayTeam'].get('score', 0)) if self.selected_match['state'] == 'in' else 0
            
            prediction = self.ml_engine.predict_match(h_name, a_name, iterations=iterations, progress_callback=self.update_progress, live_home_score=live_h, live_away_score=live_a)
            
            self.after(0, lambda p=prediction: self.show_results(p))
            
        threading.Thread(target=run_ml, daemon=True).start()
        
    def clear_training(self):
        self.is_training = False
        self.progress_bar.set(0)
        self.progress_label.configure(text="Scenarios Simulated: 0")
        if self.selected_match and self.selected_match['state'] != 'post':
            self.status_label.configure(text="")
            self.train_btn.configure(state="normal", text="Run XGBoost Regressor")
        self.results_frame.grid_forget()
        
    def show_results(self, prediction):
        self.is_training = False
        self.train_btn.configure(state="normal", text="Re-Simulate")
        
        if not prediction or "error" in prediction:
            msg = prediction.get("error", "Error generating prediction.") if prediction else "Error generating prediction."
            self.status_label.configure(text=msg, text_color="#EF4444")
            return
            
        self.results_frame.grid(row=6, column=0, pady=20, padx=40, sticky="ew")
        self.results_frame.grid_columnconfigure((0,1,2), weight=1)
        
        self.results_label.grid(row=0, column=0, columnspan=3, pady=(20, 20))
        
        home = self.selected_match['homeTeam']['name']
        away = self.selected_match['awayTeam']['name']
        
        self.home_win_lbl.configure(text=f"{home} Win\n{prediction['home_win']}%\n(xG: {prediction['home_xg']})", font=ctk.CTkFont(size=20, weight="bold"), text_color="#3B82F6")
        self.home_win_lbl.grid(row=1, column=0, pady=10)
        
        self.draw_lbl.configure(text=f"Draw\n{prediction['draw']}%", font=ctk.CTkFont(size=20, weight="bold"), text_color="#9CA3AF")
        self.draw_lbl.grid(row=1, column=1, pady=10)
        
        self.away_win_lbl.configure(text=f"{away} Win\n{prediction['away_win']}%\n(xG: {prediction['away_xg']})", font=ctk.CTkFont(size=20, weight="bold"), text_color="#EF4444")
        self.away_win_lbl.grid(row=1, column=2, pady=10)
        
        scores_str = ""
        all_scores_str = ""
        for idx, (score, pct) in enumerate(prediction['scorelines']):
            if idx < 9:
                scores_str += f"{score} ({pct}%)        "
                if (idx + 1) % 3 == 0:
                    scores_str = scores_str.strip() + "\n\n"
                    
            all_scores_str += f"Score: {score}  |  Probability: {pct}%\n"
            
        self.scorelines_lbl.configure(text=scores_str.strip(), text_color="#FBBF24", font=ctk.CTkFont(size=20, weight="bold"))
        self.scorelines_lbl.grid(row=2, column=0, columnspan=3, pady=(20, 10))
        
        self.all_scores_lbl.configure(text=all_scores_str.strip())
        self.show_all_btn.grid(row=3, column=0, columnspan=3, pady=(0, 20))

        self.is_showing_all = False
        self.show_all_btn.configure(text="Show All Probabilities ▼")
        self.all_scores_frame.grid_forget()
        
    def toggle_all_scores(self):
        self.is_showing_all = not self.is_showing_all
        if self.is_showing_all:
            self.show_all_btn.configure(text="Hide Probabilities ▲")
            self.all_scores_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=20, pady=(0, 20))
        else:
            self.show_all_btn.configure(text="Show All Probabilities ▼")
            self.all_scores_frame.grid_forget()

if __name__ == "__main__":
    app = WorldCupMLApp()
    app.mainloop()
