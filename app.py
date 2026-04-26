import streamlit as st
import pandas as pd
import requests
import pulp
import json
import concurrent.futures
import numpy as np
from scipy.stats import poisson
import google.generativeai as genai

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="FPL AI Manager", page_icon="⚽", layout="wide")
st.title("🤖 Ultimátní FPL AI Manager (Verze 2.0 - Hybridní Model)")

# --- INICIALIZACE PAMĚTI (Session State) ---
if 'my_team' not in st.session_state:
    st.session_state['my_team'] = []
if 'bank' not in st.session_state:
    st.session_state['bank'] = 0.0
if 'nlp_modifiers' not in st.session_state:
    st.session_state['nlp_modifiers'] = []

# --- 1. POKROČILÁ MATEMATIKA (DEEP RESEARCH) ---
def calc_ema(series, alpha=0.25):
    if not series: return 0.0
    ema = series[0]
    for val in series[1:]:
        ema = alpha * val + (1 - alpha) * ema
    return ema

def calculate_advanced_xpts(pos, mins, xg, xa, xgc, cs, saves, cbit, cbirt):
    if mins == 0: return 0.0
    p90 = mins / 90.0
    
    xg_90 = float(xg) / p90
    xa_90 = float(xa) / p90
    xgc_90 = float(xgc) / p90
    cs_90 = float(cs) / p90
    saves_90 = float(saves) / p90
    cbit_90 = float(cbit) / p90
    cbirt_90 = float(cbirt) / p90

    base_pts = 2.0 
    
    if pos == 'DEF':
        p_cbit_bonus = poisson.sf(9, cbit_90) 
        def_bonus = p_cbit_bonus * 2.0
    else:
        p_cbirt_bonus = poisson.sf(11, cbirt_90)
        def_bonus = p_cbirt_bonus * 2.0

    if pos == 'GK':
        return base_pts + (saves_90 * 0.33) + (cs_90 * 4.0) - (xgc_90 * 0.5)
    elif pos == 'DEF':
        return base_pts + (xg_90 * 6.0) + (xa_90 * 3.0) + (cs_90 * 4.0) - (xgc_90 * 0.5) + def_bonus
    elif pos == 'MID':
        return base_pts + (xg_90 * 5.0) + (xa_90 * 3.0) + (cs_90 * 1.0) + def_bonus
    elif pos == 'FWD':
        return base_pts + (xg_90 * 4.0) + (xa_90 * 3.0) + def_bonus
    return 0.0

# --- 2. DATOVÁ ČÁST A TVORBA KURZŮ ---
@st.cache_data(ttl=3600)
def get_current_gw():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    res = requests.get(url).json()
    for event in res['events']:
        if event['is_current']: return event['id']
    for event in res['events']:
        if event['is_previous']: return event['id']
    return 1

def fetch_player_history(player_id):
    url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
    res = requests.get(url).json()
    return player_id, res['history']

@st.cache_data(ttl=3600)
def load_fpl_data():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url).json()
    
    df = pd.DataFrame(response['elements'])
    teams_df = pd.DataFrame(response['teams'])
    
    team_mapping = dict(zip(teams_df['id'], teams_df['name']))
    team_short_mapping = dict(zip(teams_df['id'], teams_df['short_name']))
    
    df['team_name'] = df['team'].map(team_mapping)
    df['unique_name'] = df['web_name'] + " (" + df['team_name'] + ")"
    df['now_cost'] = df['now_cost'] / 10.0
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    df['position'] = df['element_type'].map(position_map)
    
    active_players = df[df['minutes'] > 100]['id'].tolist()
    history_data = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_player_history, pid): pid for pid in active_players}
        for future in concurrent.futures.as_completed(futures):
            pid, hist = future.result()
            history_data[pid] = hist

    custom_forms = []
    for idx, row in df.iterrows():
        pid = row['id']
        pos = row['position']
        
        if pid not in history_data or row['minutes'] == 0:
            custom_forms.append(0.0)
            continue
            
        hist = history_data[pid]
        match_xpts = []
        
        for m in hist:
            if m['minutes'] > 0:
                cbi = m.get('clearances_blocks_interceptions', 0)
                tackles = m.get('tackles', 0)
                recoveries = m.get('recoveries', 0)
                cbit = cbi + tackles
                cbirt = cbit + recoveries
                
                pts = calculate_advanced_xpts(
                    pos, m['minutes'], m['expected_goals'], m['expected_assists'], 
                    m['expected_goals_conceded'], m['clean_sheets'], m['saves'], cbit, cbirt
                )
                match_xpts.append(pts)
        
        ema_form = calc_ema(match_xpts, alpha=0.25) if match_xpts else 0.0
        recent_5 = hist[-5:]
        starts = sum(1 for m in recent_5 if m['minutes'] > 45)
        poa = starts / 5.0 if len(recent_5) == 5 else (starts / len(recent_5) if len(recent_5) > 0 else 0)
        
        final_form = ema_form * poa
        custom_forms.append(max(0.0, final_form))

    df['form'] = custom_forms
    
    df['chance_of_playing_next_round'] = pd.to_numeric(df['chance_of_playing_next_round'], errors='coerce').fillna(100)
    df['health_multiplier'] = df['chance_of_playing_next_round'] / 100.0
    df['news'] = df['news'].fillna('')
    
    fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/?future=1'
    fixtures_data = requests.get(fixtures_url).json()
    
    team_fdr_5gw = {team_id: [] for team_id in teams_df['id']}
    team_fdr_1gw = {team_id: None for team_id in teams_df['id']}
    team_fixtures_str = {team_id: [] for team_id in teams_df['id']}
    team_fixtures_diff = {team_id: [] for team_id in teams_df['id']}
    
    for f in fixtures_data:
        team_h = f['team_h']
        team_a = f['team_a']
        diff_h = f['team_h_difficulty']
        diff_a = f['team_a_difficulty']
        
        if len(team_fdr_5gw[team_h]) < 5:
            team_fdr_5gw[team_h].append(diff_h)
            team_fixtures_str[team_h].append(f"{team_short_mapping[team_a]} (H)")
            team_fixtures_diff[team_h].append(diff_h)
            
        if len(team_fdr_5gw[team_a]) < 5:
            team_fdr_5gw[team_a].append(diff_a)
            team_fixtures_str[team_a].append(f"{team_short_mapping[team_h]} (A)")
            team_fixtures_diff[team_a].append(diff_a)
            
        if team_fdr_1gw[team_h] is None: team_fdr_1gw[team_h] = diff_h
        if team_fdr_1gw[team_a] is None: team_fdr_1gw[team_a] = diff_a
            
    team_multiplier_5gw = {}
    team_multiplier_1gw = {}
    
    for team_id in teams_df['id']:
        fdrs_5 = team_fdr_5gw[team_id]
        avg_fdr_5 = sum(fdrs_5) / len(fdrs_5) if len(fdrs_5) > 0 else 3.0
        team_multiplier_5gw[team_id] = round(1.0 + (3.0 - avg_fdr_5) * 0.2, 2)
        
        fdr_1 = team_fdr_1gw[team_id] if team_fdr_1gw[team_id] is not None else 3.0
        team_multiplier_1gw[team_id] = round(1.0 + (3.0 - fdr_1) * 0.2, 2)
        
    df['fdr_multiplier_5gw'] = df['team'].map(team_multiplier_5gw)
    df['fdr_multiplier_1gw'] = df['team'].map(team_multiplier_1gw)
    
    # ZÁKLADNÍ MODEL (Čistě naše data)
    df['model_1gw_fdr'] = (df['form'] * 1) * df['fdr_multiplier_1gw'] * df['health_multiplier']
    df['projected_5gw_fdr'] = (df['form'] * 5) * df['fdr_multiplier_5gw'] * df['health_multiplier']
    
    # --- NOVINKA: SIMULACE SÁZKOVÝCH KURZŮ (Odds Compiler) ---
    cs_odds_map = {1: 1.80, 2: 2.20, 3: 3.50, 4: 5.50, 5: 8.00} 
    
    odds_goal = []
    odds_cs = []
    odds_implied_pts = []
    
    for idx, row in df.iterrows():
        fdr_next = team_fdr_1gw[row['team']] if team_fdr_1gw[row['team']] is not None else 3
        
        cs_odd = cs_odds_map.get(fdr_next, 3.50)
        cs_prob = 1.0 / cs_odd
        odds_cs.append(cs_odd)
        
        base_goal_prob = (row['form'] / 15.0) * (1.0 + (3.0 - fdr_next) * 0.15)
        
        if row['position'] == 'FWD': goal_prob = min(0.65, max(0.05, base_goal_prob))
        elif row['position'] == 'MID': goal_prob = min(0.45, max(0.02, base_goal_prob * 0.7))
        elif row['position'] == 'DEF': goal_prob = min(0.15, max(0.01, base_goal_prob * 0.2))
        else: goal_prob = 0.001
        
        goal_odd = round(1.0 / goal_prob, 2) if goal_prob > 0 else 99.0
        odds_goal.append(goal_odd)
        
        pts_from_odds = 2.0 
        if row['position'] in ['DEF', 'GK']: pts_from_odds += (cs_prob * 4.0)
        elif row['position'] == 'MID': pts_from_odds += (cs_prob * 1.0)
        
        if row['position'] == 'FWD': pts_from_odds += (goal_prob * 4.0)
        elif row['position'] == 'MID': pts_from_odds += (goal_prob * 5.0)
        elif row['position'] == 'DEF': pts_from_odds += (goal_prob * 6.0)
        
        pts_from_odds += (row['form'] * 0.3) 
        
        odds_implied_pts.append(pts_from_odds * row['health_multiplier'])

    df['odds_goal'] = odds_goal
    df['odds_cs'] = odds_cs
    df['odds_1gw_pts'] = odds_implied_pts
    
    for i in range(5):
        df[f'Zápas {i+1}'] = df['team'].apply(lambda x: team_fixtures_str[x][i] if len(team_fixtures_str[x]) > i else "-")
        df[f'Diff {i+1}'] = df['team'].apply(lambda x: team_fixtures_diff[x][i] if len(team_fixtures_diff[x]) > i else 3)
    
    return df

def fetch_manager_team(manager_id, current_gw, df):
    url = f'https://fantasy.premierleague.com/api/entry/{manager_id}/event/{current_gw}/picks/'
    res = requests.get(url)
    if res.status_code != 200:
        return None, None, current_gw
    
    data = res.json()
    if data.get('active_chip') == 'freehit':
        return fetch_manager_team(manager_id, current_gw - 1, df)
        
    player_ids = [pick['element'] for pick in data['picks']]
    team_names = df[df['id'].isin(player_ids)]['unique_name'].tolist()
    bank = data['entry_history']['bank'] / 10.0
    return team_names, bank, current_gw

def get_best_xi(squad_df):
    squad = squad_df.sort_values(by='projected_1gw_fdr', ascending=False)
    
    gk = squad[squad['position'] == 'GK']
    df = squad[squad['position'] == 'DEF']
    md = squad[squad['position'] == 'MID']
    fw = squad[squad['position'] == 'FWD']
    
    start_idx = [
        gk.index[0],
        df.index[0], df.index[1], df.index[2],
        md.index[0], md.index[1],
        fw.index[0]
    ]
    
    remaining_idx = [idx for idx in squad.index if idx not in start_idx and idx != gk.index[1]]
    remaining_players = squad.loc[remaining_idx].sort_values(by='projected_1gw_fdr', ascending=False)
    
    start_idx.extend(remaining_players.index[:4])
    bench_idx = [gk.index[1]] + list(remaining_players.index[4:])
    
    start_df = squad.loc[start_idx].sort_values(by='projected_1gw_fdr', ascending=False)
    captain_id = start_df.iloc[0]['id']
    vc_id = start_df.iloc[1]['id']
    
    start_df['pos_order'] = start_df['position'].map({'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    start_df = start_df.sort_values(by=['pos_order', 'projected_1gw_fdr'], ascending=[True, False])
    
    bench_df = squad.loc[bench_idx]
    
    captain_points = start_df.loc[start_df['id'] == captain_id, 'projected_1gw_fdr'].values[0]
    total_xi_points = start_df['projected_1gw_fdr'].sum() + captain_points
    
    return start_df, bench_df, captain_id, vc_id, total_xi_points

# Načtení dat
with st.spinner("Stahuji data a počítám stochastický model formy (EMA + Poisson)..."):
    df = load_fpl_data()

# --- APLIKACE NLP MODIFIKÁTORŮ Z TISKOVEK ---
if st.session_state['nlp_modifiers']:
    for mod in st.session_state['nlp_modifiers']:
        idx = df['web_name'] == mod['web_name']
        df.loc[idx, 'projected_5gw_fdr'] *= mod['xMins_multiplier']
        df.loc[idx, 'model_1gw_fdr'] *= mod['xMins_multiplier']
        df.loc[idx, 'odds_1gw_pts'] *= mod['xMins_multiplier']

# --- 3. BOČNÍ PANEL ---
st.sidebar.header("📥 Import týmu")
manager_id = st.sidebar.text_input("Zadej své FPL ID (např. 123456):")

if st.sidebar.button("⬇️ Stáhnout můj tým", type="primary"):
    if manager_id.isdigit():
        with st.spinner("Stahuji data z FPL..."):
            gw = get_current_gw()
            fetched_team, fetched_bank, real_gw = fetch_manager_team(manager_id, gw, df)
            
            if fetched_team and len(fetched_team) == 15:
                st.session_state['my_team'] = fetched_team
                st.session_state['bank'] = fetched_bank
                if real_gw < gw:
                    st.sidebar.warning(f"⚠️ Detekován Free Hit v GW{gw}! Načten tvůj permanentní tým z GW{real_gw}.")
                else:
                    st.sidebar.success(f"✅ Tým úspěšně načten z Gameweeku {gw}!")
            else:
                st.sidebar.error("❌ Nepodařilo se načíst tým.")
    else:
        st.sidebar.error("⚠️ ID musí obsahovat pouze čísla.")

st.sidebar.divider()

# --- NOVINKA: HYBRIDNÍ MODEL (Váha kurzů) ---
st.sidebar.header("🎲 Hybridní Model (Kurzy)")
odds_weight = st.sidebar.slider("Váha sázkových kurzů v projekci:", min_value=0, max_value=100, value=50, step=10, help="0% = Pouze náš matematický model. 100% = Pouze sázkové kurzy. 50% = Ideální mix obojího.")
odds_ratio = odds_weight / 100.0

# Aplikace hybridního modelu do hlavní projekce
df['projected_1gw_fdr'] = (df['model_1gw_fdr'] * (1.0 - odds_ratio)) + (df['odds_1gw_pts'] * odds_ratio)

st.sidebar.divider()

st.sidebar.header("🔄 Správa dat")
if st.sidebar.button("Vynutit přepočet surových dat (CBIT/CBIRT)"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.header("⚙️ Tvoje nastavení")
bank = st.sidebar.number_input("Peníze v bance (miliony):", min_value=0.0, max_value=100.0, value=float(st.session_state['bank']), step=0.1)
free_transfers = st.sidebar.slider("Počet volných přestupů:", 1, 5, 1)

st.sidebar.subheader("Tvůj aktuální tým")
all_player_names = sorted(df['unique_name'].tolist())
valid_team = [name for name in st.session_state['my_team'] if name in all_player_names]
my_team = st.sidebar.multiselect("Vyber přesně 15 hráčů:", all_player_names, default=valid_team, max_selections=15)

if st.session_state['nlp_modifiers']:
    st.sidebar.divider()
    st.sidebar.subheader("🏥 Aktivní AI hlášení z tiskovek")
    for mod in st.session_state['nlp_modifiers']:
        if mod['xMins_multiplier'] < 1.0:
            st.sidebar.error(f"**{mod['web_name']}**: {mod['reason']}")

# --- 4. HLAVNÍ OBSAH ---
tab1, tab2, tab3 = st.tabs(["🔄 Optimalizátor přestupů", "📅 Databáze & Kurzy", "🧠 AI Analýza tiskovek"])

with tab1:
    st.header("Matematický návrh přestupů")
    
    if len(my_team) == 15:
        if st.button("🚀 Spustit AI Optimalizaci", type="primary"):
            with st.spinner('AI prohledává miliony kombinací a skládá základní sestavu...'):
                current_squad_ids = df[df['unique_name'].isin(my_team)]['id'].tolist()
                current_squad_df = df[df['id'].isin(current_squad_ids)]
                total_budget = current_squad_df['now_cost'].sum() + bank
                
                current_squad_5gw_proj = current_squad_df['projected_5gw_fdr'].sum()

                prob = pulp.LpProblem("FPL_Transfer_Optimizer", pulp.LpMaximize)
                player_vars = pulp.LpVariable.dicts("player", df['id'], cat='Binary')
                
                projections = dict(zip(df['id'], df['projected_5gw_fdr']))
                costs = dict(zip(df['id'], df['now_cost']))
                
                prob += pulp.lpSum([projections[i] * player_vars[i] for i in df['id']])
                prob += pulp.lpSum([player_vars[i] for i in df['id']]) == 15
                prob += pulp.lpSum([costs[i] * player_vars[i] for i in df['id']]) <= total_budget
                prob += pulp.lpSum([player_vars[i] for i in df[df['position'] == 'GK']['id']]) == 2
                prob += pulp.lpSum([player_vars[i] for i in df[df['position'] == 'DEF']['id']]) == 5
                prob += pulp.lpSum([player_vars[i] for i in df[df['position'] == 'MID']['id']]) == 5
                prob += pulp.lpSum([player_vars[i] for i in df[df['position'] == 'FWD']['id']]) == 3
                
                for team in df['team_name'].unique():
                    prob += pulp.lpSum([player_vars[i] for i in df[df['team_name'] == team]['id']]) <= 3

                prob += pulp.lpSum([player_vars[i] for i in current_squad_ids]) >= (15 - free_transfers)

                prob.solve()
                
                if pulp.LpStatus[prob.status] == 'Optimal':
                    selected_ids = [i for i in df['id'] if player_vars[i].varValue == 1]
                    new_squad_df = df[df['id'].isin(selected_ids)]
                    
                    new_squad_5gw_proj = new_squad_df['projected_5gw_fdr'].sum()
                    new_start, new_bench, cap_id, vc_id, new_xi_1gw_proj = get_best_xi(new_squad_df)
                    
                    players_out = current_squad_df[~current_squad_df['id'].isin(selected_ids)]
                    players_in = new_squad_df[~new_squad_df['id'].isin(current_squad_ids)]
                    
                    st.subheader("🔄 Doporučené přestupy:")
                    col1, col2 = st.columns(2)
                    with col1:
                        for _, p_out in players_out.iterrows():
                            st.error(f"❌ PRODEJ: {p_out['unique_name']} ({p_out['now_cost']}m) | Projekce: {p_out['projected_5gw_fdr']:.1f} b.")
                    with col2:
                        for _, p_in in players_in.iterrows():
                            st.success(f"✅ KUP: {p_in['unique_name']} ({p_in['now_cost']}m) | Projekce: {p_in['projected_5gw_fdr']:.1f} b.")
                            
                    st.divider()
                    
                    st.subheader("🏟️ Vizuální hřiště (Příští kolo)")
                    for pos in ['GK', 'DEF', 'MID', 'FWD']:
                        players_in_pos = new_start[new_start['position'] == pos]
                        if not players_in_pos.empty:
                            cols = st.columns(len(players_in_pos))
                            for col, (_, row) in zip(cols, players_in_pos.iterrows()):
                                with col:
                                    role = ""
                                    if row['id'] == cap_id: role = " <span style='color: #FFD700;'>**(C)**</span>"
                                    elif row['id'] == vc_id: role = " *(VC)*"
                                    
                                    health_icon = f" <span title='{row['news']
}' style='cursor: help;'>🏥</span>" if row['health_multiplier'] < 1.0 else ""
                                        
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 10px; background-color: rgba(150, 150, 150, 0.1); border-radius: 10px; border: 1px solid rgba(150, 150, 150, 0.2); margin-bottom: 10px;">
                                        <div style="font-size: 28px;">👕</div>
                                        <div style="font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{row['web_name']}">{row['web_name']}{role}{health_icon}</div>
                                        <div style="font-size: 14px; color: #4CAF50; font-weight: bold;">{row['projected_1gw_fdr']:.1f} b.</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("🪑 Lavička náhradníků")
                    bench_cols = st.columns(4)
                    for col, (_, row) in zip(bench_cols, new_bench.iterrows()):
                        with col:
                            health_icon = f" <span title='{row['news']}' style='cursor: help;'>🏥</span>" if row['health_multiplier'] < 1.0 else ""
                            st.markdown(f"""
                            <div style="text-align: center; padding: 8px; background-color: rgba(255, 99, 71, 0.1); border-radius: 10px; border: 1px dashed rgba(255, 99, 71, 0.3);">
                                <div style="font-weight: bold; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['web_name']}{health_icon}</div>
                                <div style="font-size: 12px; color: gray;">{row['position']} | {row['projected_1gw_fdr']:.1f} b.</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.divider()
                    col_met1, col_met2, col_met3 = st.columns(3)
                    col_met1.metric("Zisk z přestupu (Celý tým na 5 kol)", f"+{new_squad_5gw_proj - current_squad_5gw_proj:.1f} bodů")
                    col_met2.metric("Očekávané body sestavy (Příští kolo)", f"{new_xi_1gw_proj:.1f} bodů")
                    col_met3.metric("Zůstatek v bance", f"{total_budget - new_squad_df['now_cost'].sum():.1f} m")
                else:
                    st.error("Nepodařilo se najít řešení. Zkontroluj rozpočet.")
    else:
        st.info(f"👈 Vyber v levém panelu přesně 15 hráčů. Zatím jich máš {len(my_team)}.")

with tab2:
    st.header("Kompletní databáze hráčů a Sázkové kurzy")
    
    all_teams = sorted(df['team_name'].unique().tolist())
    selected_teams = st.multiselect("🔍 Filtrovat podle klubů:", all_teams, default=[], placeholder="Vyber jeden nebo více týmů...")
    
    if selected_teams:
        filtered_df = df[df['team_name'].isin(selected_teams)]
    else:
        filtered_df = df
    
    display_df = filtered_df[['unique_name', 'position', 'now_cost', 'odds_goal', 'odds_cs', 'projected_1gw_fdr', 'projected_5gw_fdr', 'Zápas 1', 'Zápas 2', 'Zápas 3', 'Zápas 4', 'Zápas 5']].copy()
    display_df.columns = ['Hráč (Tým)', 'Pozice', 'Cena', 'Kurz na Gól', 'Kurz na ČK', 'Hybridní Projekce (1 kolo)', 'Projekce (5 kol)', 'Zápas 1', 'Zápas 2', 'Zápas 3', 'Zápas 4', 'Zápas 5']
    
    diff_df = filtered_df[['Diff 1', 'Diff 2', 'Diff 3', 'Diff 4', 'Diff 5']].copy()
    diff_df.columns = ['Zápas 1', 'Zápas 2', 'Zápas 3', 'Zápas 4', 'Zápas 5']
    
    def style_fixtures(data, diffs):
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        for col in ['Zápas 1', 'Zápas 2', 'Zápas 3', 'Zápas 4', 'Zápas 5']:
            for idx in data.index:
                val = diffs.loc[idx, col]
                if val == 1: bg, text = '#006400', 'white'
                elif val == 2: bg, text = '#2cba00', 'white'
                elif val == 3: bg, text = '#a9a9a9', 'black'
                elif val == 4: bg, text = '#ff4e11', 'white'
                elif val == 5: bg, text = '#8B0000', 'white'
                else: bg, text = '', ''
                styles.loc[idx, col] = f'background-color: {bg}; color: {text}; text-align: center; font-weight: bold;'
        return styles

    styled_df = display_df.style.apply(style_fixtures, diffs=diff_df, axis=None).format({
        'Cena': "{:.1f}", 'Kurz na Gól': "{:.2f}", 'Kurz na ČK': "{:.2f}", 'Hybridní Projekce (1 kolo)': "{:.1f}", 'Projekce (5 kol)': "{:.1f}"
    })
    
    st.dataframe(styled_df, use_container_width=True, height=600)

with tab3:
    st.header("🧠 AI Analýza tiskových konferencí (Google Gemini)")
    st.write("Vlož text z tiskovky. AI z něj extrahuje zranění a automaticky upraví projekce hráčů v celém systému!")
    
    api_key = st.text_input("Zadej svůj Google Gemini API klíč (začíná na AIza...):", type="password", help="Získáš ho ZDARMA na aistudio.google.com")
    news_text = st.text_area("Text z tiskovky (nebo novinky z Twitteru):", height=200, placeholder="Např.: Haaland si poranil hamstring a o víkendu nenastoupí. Foden je unavený a začne na lavičce...")
    
    if st.button("🧠 Analyzovat text a upravit projekce", type="primary"):
        if not news_text:
            st.warning("Nejprve vlož nějaký text.")
        else:
            with st.spinner("AI čte text a hledá zranění..."):
                try:
                    if api_key == "":
                        st.warning("Nebyl zadán API klíč. Používám simulovaná (demo) data pro ukázku...")
                        demo_json = """
                        {
                            "players": [
                                {"web_name": "Haaland", "xMins_multiplier": 0.0, "reason": "Demo: Zraněný hamstring, nehraje."},
                                {"web_name": "Foden", "xMins_multiplier": 0.25, "reason": "Demo: Unavený, začne na lavičce."}
                            ]
                        }
                        """
                        extracted_data = json.loads(demo_json)['players']
                    else:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
                        prompt = """
                        Jsi expert na Fantasy Premier League. Přečti si text.
                        Vrať POUZE validní JSON ve formátu:
                        {"players": [{"web_name": "Jméno", "xMins_multiplier": 0.0, "reason": "Důvod"}]}
                        xMins_multiplier je 0.0 (nehraje), 0.25 (lavička), 0.75 (střídá), 1.0 (hraje).
                        """
                        response = model.generate_content(f"{prompt}\n\nText k analýze:\n{news_text}")
                        extracted_data = json.loads(response.text)['players']
                    
                    st.session_state['nlp_modifiers'] = extracted_data
                    st.success("✅ Analýza dokončena! Projekce hráčů byly upraveny.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Nastala chyba při komunikaci s AI: {e}")
