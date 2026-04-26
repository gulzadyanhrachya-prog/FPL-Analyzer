import streamlit as st
import pandas as pd
import requests
import pulp
import json
import concurrent.futures
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import google.generativeai as genai

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="FPL AI Manager", page_icon="⚽", layout="wide")
st.title("🤖 Ultimátní FPL AI Manager (Verze 9.0 - Rival Tracker)")

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
    
    df['transfers_in'] = pd.to_numeric(df['transfers_in_event'], errors='coerce').fillna(0)
    df['transfers_out'] = pd.to_numeric(df['transfers_out_event'], errors='coerce').fillna(0)
    df['net_transfers'] = df['transfers_in'] - df['transfers_out']
    
    def predict_price_change(net_t):
        if net_t > 60000: return '📈 Roste'
        elif net_t > 20000: return '↗️ Mírný růst'
        elif net_t < -60000: return '📉 Klesá'
        elif net_t < -20000: return '↘️ Mírný pokles'
        else: return '➖ Stabilní'
        
    df['price_trend'] = df['net_transfers'].apply(predict_price_change)
    
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
    
    df['model_1gw_fdr'] = (df['form'] * 1) * df['fdr_multiplier_1gw'] * df['health_multiplier']
    df['projected_5gw_fdr'] = (df['form'] * 5) * df['fdr_multiplier_5gw'] * df['health_multiplier']
    
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
    
    df['xG_90'] = pd.to_numeric(df['expected_goals'], errors='coerce').fillna(0) / np.maximum(df['minutes'], 1) * 90
    df['xA_90'] = pd.to_numeric(df['expected_assists'], errors='coerce').fillna(0) / np.maximum(df['minutes'], 1) * 90
    df['Goal_Prob'] = 100.0 / df['odds_goal']
    df['CS_Prob'] = 100.0 / df['odds_cs']
    
    df['xG_pct'] = df['xG_90'].rank(pct=True) * 100
    df['xA_pct'] = df['xA_90'].rank(pct=True) * 100
    df['Form_pct'] = df['form'].rank(pct=True) * 100
    df['Proj_pct'] = df['model_1gw_fdr'].rank(pct=True) * 100
    df['Goal_pct'] = df['Goal_Prob'].rank(pct=True) * 100
    df['CS_pct'] = df['CS_Prob'].rank(pct=True) * 100
    
    for i in range(5):
        df[f'Zápas {i+1}'] = df['team'].apply(lambda x: team_fixtures_str[x][i] if len(team_fixtures_str[x]) > i else "-")
        df[f'Diff {i+1}'] = df['team'].apply(lambda x: team_fixtures_diff[x][i] if len(team_fixtures_diff[x]) > i else 3)
        
        fdr_mult = 1.0 + (3.0 - df[f'Diff {i+1}']) * 0.2
        df[f'proj_gw{i+1}'] = df['form'] * fdr_mult * df['health_multiplier']
    
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

# --- NOVÉ FUNKCE PRO LIVE TRACKER A MINI-LIGY ---
def fetch_live_manager_data(manager_id, gw):
    url = f'https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gw}/picks/'
    res = requests.get(url)
    if res.status_code != 200: return None
    return res.json()

def fetch_live_event_data(gw):
    url = f'https://fantasy.premierleague.com/api/event/{gw}/live/'
    res = requests.get(url)
    if res.status_code != 200: return {}
    data = res.json()
    return {item['id']: item['stats'] for item in data.get('elements', [])}

def get_best_xi(squad_df):
    squad = squad_df.sort_values(by='projected_1gw_fdr', ascending=False)
    
    gk = squad[squad['position'] == 'GK']
    df_pos = squad[squad['position'] == 'DEF']
    md = squad[squad['position'] == 'MID']
    fw = squad[squad['position'] == 'FWD']
    
    start_idx = [
        gk.index[0],
        df_pos.index[0], df_pos.index[1], df_pos.index[2],
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
        for i in range(1, 6):
            df.loc[idx, f'proj_gw{i}'] *= mod['xMins_multiplier']

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

# --- VŠECHNY ŽOLÍKY (CHIPS) ---
st.sidebar.header("🃏 Strategie žolíků")
active_chip = st.sidebar.radio(
    "Aktivovat čip pro příští kolo:", 
    ["Žádný", "🃏 Wildcard", "🆓 Free Hit", "🚀 Bench Boost"],
    help="Wildcard a Free Hit zruší limit přestupů. Free Hit optimalizuje jen na 1 kolo. Bench Boost započítá lavičku."
)

simulate_wildcard = (active_chip == "🃏 Wildcard")
simulate_freehit = (active_chip == "🆓 Free Hit")
simulate_bb = (active_chip == "🚀 Bench Boost")

st.sidebar.divider()

st.sidebar.header("🎲 Hybridní Model (Kurzy)")
odds_weight = st.sidebar.slider("Váha sázkových kurzů v projekci:", min_value=0, max_value=100, value=50, step=10)
odds_ratio = odds_weight / 100.0

df['projected_1gw_fdr'] = (df['model_1gw_fdr'] * (1.0 - odds_ratio)) + (df['odds_1gw_pts'] * odds_ratio)
df['proj_gw1'] = df['projected_1gw_fdr']

st.sidebar.divider()

st.sidebar.header("🔄 Správa dat")
if st.sidebar.button("Vynutit přepočet surových dat (CBIT/CBIRT)"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.header("⚙️ Tvoje nastavení")
bank = st.sidebar.number_input("Peníze v bance (miliony):", min_value=0.0, max_value=100.0, value=float(st.session_state['bank']), step=0.1)
free_transfers = st.sidebar.slider("Počet volných přestupů:", 1, 5, 1, disabled=(simulate_wildcard or simulate_freehit))

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
# PŘIDÁNA NOVÁ ZÁLOŽKA PRO MINI-LIGY (TAB LEAGUE)
tab_home, tab_live, tab_league, tab1, tab5, tab6, tab2, tab3, tab4 = st.tabs(["🏠 Hlavní Dashboard", "🔴 Live Gameweek", "⚔️ Mini-Ligy", "🔄 Rychlý Optimalizátor", "🚀 Vícekolový plánovač", "©️ Plánovač Kapitánů", "📅 Databáze & Kurzy", "🕸️ Porovnávač hráčů", "🧠 AI Analýza tiskovek"])

# --- HLAVNÍ DASHBOARD ---
with tab_home:
    st.header("🏠 Hlavní Dashboard manažera")
    
    if len(my_team) == 15:
        current_squad_ids = df[df['unique_name'].isin(my_team)]['id'].tolist()
        current_squad_df = df[df['id'].isin(current_squad_ids)]
        
        c_start, c_bench, c_cap, c_vc, c_xi_pts = get_best_xi(current_squad_df)
        team_value = current_squad_df['now_cost'].sum() + bank
        
        st.subheader("📊 Rychlý přehled")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hodnota týmu", f"{team_value:.1f} m")
        col2.metric("V bance", f"{bank:.1f} m")
        col3.metric("Očekávané body (Příští kolo)", f"{c_xi_pts:.1f} b.")
        col4.metric("Volné přestupy", f"{free_transfers}")
        
        st.divider()
        
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            st.subheader("🏥 AI Zdravotní prohlídka")
            injured = current_squad_df[current_squad_df['health_multiplier'] < 1.0]
            if injured.empty:
                st.success("✅ Všichni hráči ve tvém týmu jsou plně fit a připraveni hrát!")
            else:
                for _, row in injured.iterrows():
                    st.error(f"**{row['web_name']}**: {row['news']} (Šance na start: {int(row['chance_of_playing_next_round'])}%)")
                    
        with col_b:
            st.subheader("📅 Analýza losu (FDR)")
            avg_fdr = current_squad_df[['Diff 1', 'Diff 2', 'Diff 3', 'Diff 4', 'Diff 5']].mean().mean()
            st.metric("Průměrná náročnost losu (1-5, menší je lepší)", f"{avg_fdr:.2f}")
            
            if avg_fdr < 2.6:
                st.success("🔥 Tvůj tým má před sebou skvělý los! Ideální čas na útok v žebříčku.")
            elif avg_fdr > 3.2:
                st.warning("⚠️ Tvůj tým čeká těžká série zápasů. Zvaž Wildcard nebo cílené přestupy.")
            else:
                st.info("⚖️ Los tvého týmu je průměrný. Zaměř se na formu jednotlivců.")
                
        st.divider()
        st.subheader("⭐ Hvězdy týmu (Top 3 hráči ve formě)")
        top_form = current_squad_df.sort_values(by='form', ascending=False).head(3)
        cols = st.columns(3)
        for col, (_, row) in zip(cols, top_form.iterrows()):
            with col:
                st.markdown(f"**{row['web_name']}**")
                st.progress(min(row['Form_pct']/100.0, 1.0), text=f"Forma: {row['form']:.1f} (Lepší než {int(row['Form_pct'])}% ligy)")
                
    else:
        st.info("👋 Vítej v Ultimátním FPL AI Managerovi!")
        st.write("Tento nástroj kombinuje stochastickou matematiku, lineární programování a umělou inteligenci od Googlu, aby ti pomohl vyhrát tvou mini-ligu.")
        st.write("👉 **Pro zobrazení dashboardu si v levém panelu stáhni svůj tým (zadej FPL ID) nebo ručně vyber 15 hráčů.**")

# --- LIVE GAMEWEEK TRACKER ---
with tab_live:
    st.header("🔴 Live Gameweek Tracker")
    st.write("Sleduj své body, minuty a bonusy (BPS) v reálném čase během víkendu! Aplikace si stáhne tvé skutečné rozestavení a kapitána pro aktuální kolo.")

    if manager_id and manager_id.isdigit():
        if st.button("🔄 Aktualizovat živá data", type="primary"):
            with st.spinner("Načítám živá data z FPL serverů..."):
                gw = get_current_gw()
                live_picks_data = fetch_live_manager_data(manager_id, gw)
                live_event_data = fetch_live_event_data(gw)

                if live_picks_data and live_event_data:
                    real_active_chip = live_picks_data.get('active_chip', 'Žádný')
                    picks = live_picks_data.get('picks', [])

                    total_live_pts = 0
                    live_rows = []

                    for pick in picks:
                        pid = pick['element']
                        mult = pick['multiplier']
                        is_c = pick['is_captain']
                        is_vc = pick['is_vice_captain']

                        stats = live_event_data.get(pid, {})
                        pts = stats.get('total_points', 0)
                        mins = stats.get('minutes', 0)
                        bps = stats.get('bps', 0)

                        live_pts = pts * mult
                        total_live_pts += live_pts

                        p_match = df[df['id'] == pid]
                        p_name = p_match['web_name'].values[0] if not p_match.empty else "Neznámý"
                        p_pos = p_match['position'].values[0] if not p_match.empty else "-"

                        role = ""
                        if is_c: role = "👑 (C)"
                        elif is_vc: role = "🥈 (VC)"

                        status = "⚪ Nehrál"
                        if mins > 0: status = "🟢 Hraje / Dohrál"

                        live_rows.append({
                            "Hráč": f"{p_name} {role}",
                            "Pozice": p_pos,
                            "Status": status,
                            "Základ/Lavička": "Základ" if mult > 0 else "Lavička",
                            "Minuty": mins,
                            "BPS": bps,
                            "Živé body": live_pts
                        })

                    st.subheader(f"🏆 Aktuální skóre pro GW {gw}: {total_live_pts} bodů")
                    if real_active_chip and real_active_chip != 'Žádný':
                        st.info(f"Aktivní čip v tomto kole: {real_active_chip.upper()}")

                    live_df = pd.DataFrame(live_rows)
                    
                    def style_live_rows(row):
                        if row['Základ/Lavička'] == 'Lavička':
                            return ['color: gray; font-style: italic;'] * len(row)
                        elif '👑 (C)' in row['Hráč']:
                            return ['background-color: rgba(255, 215, 0, 0.1); font-weight: bold;'] * len(row)
                        return [''] * len(row)

                    styled_live_df = live_df.style.apply(style_live_rows, axis=1)
                    st.dataframe(styled_live_df, use_container_width=True, hide_index=True)
                else:
                    st.error("Nepodařilo se načíst živá data. Možná probíhá aktualizace FPL serverů nebo ještě nezačalo kolo.")
    else:
        st.info("👈 Zadej své FPL ID v levém panelu a stáhni svůj tým pro zobrazení Live dat.")

# --- NOVINKA: ANALYZÁTOR MINI-LIG ---
with tab_league:
    st.header("⚔️ Analyzátor Mini-lig (Rival Tracker)")
    st.write("Zadej ID své klasické mini-ligy. AI zanalyzuje týmy tvých největších rivalů (Top 10) a najde tvé největší diferenciály a hrozby!")
    
    league_id = st.text_input("🏆 Zadej ID Mini-ligy (najdeš v URL na webu FPL, např. 314):")
    
    if st.button("🔍 Analyzovat Rivaly", type="primary"):
        if league_id.isdigit():
            if len(my_team) == 15:
                with st.spinner("Stahuji data o mini-lize a analyzuji týmy rivalů..."):
                    gw = get_current_gw()
                    url_league = f'https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/'
                    res_league = requests.get(url_league)
                    
                    if res_league.status_code == 200:
                        league_data = res_league.json()
                        # Vezmeme Top 10 manažerů z ligy
                        standings = league_data.get('standings', {}).get('results', [])[:10] 
                        
                        if standings:
                            st.subheader(f"📊 Analýza Top {len(standings)} manažerů v lize: {league_data.get('league', {}).get('name', '')}")
                            
                            rival_picks = []
                            for manager in standings:
                                m_id = manager['entry']
                                # Přeskočíme tebe, pokud jsi v Top 10, abychom porovnávali jen s ostatními
                                if str(m_id) == str(manager_id):
                                    continue
                                m_team, _, _ = fetch_manager_team(m_id, gw, df)
                                if m_team:
                                    rival_picks.extend(m_team)
                                    
                            if rival_picks:
                                # Výpočet vlastnictví (Effective Ownership)
                                ownership_counts = pd.Series(rival_picks).value_counts()
                                num_rivals = len(standings) if str(manager_id) not in [str(m['entry']) for m in standings] else len(standings) - 1
                                ownership_pct = (ownership_counts / num_rivals) * 100
                                
                                my_team_set = set(my_team)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.success("🛡️ Tvoje Diferenciály (Máš je ty, ale rivalové ne)")
                                    diffs = [p for p in my_team_set if ownership_pct.get(p, 0) <= 20]
                                    if diffs:
                                        for p in diffs:
                                            st.markdown(f"- **{p}** (Vlastní jen {ownership_pct.get(p, 0):.0f} % rivalů)")
                                    else:
                                        st.info("Nemáš žádné výrazné diferenciály proti Top 10.")
                                        
                                with col2:
                                    st.error("⚠️ Největší Hrozby (Rivalové je mají, ty ne)")
                                    threats = ownership_pct[ownership_pct >= 50].index.tolist()
                                    real_threats = [p for p in threats if p not in my_team_set]
                                    if real_threats:
                                        for p in real_threats:
                                            st.markdown(f"- **{p}** (Vlastní {ownership_pct.get(p, 0):.0f} % rivalů)")
                                    else:
                                        st.success("Máš všechny klíčové hráče, které mají tvoji rivalové!")
                                        
                                st.divider()
                                st.subheader("📈 Vlastnictví hráčů v Top 10 (Effective Ownership)")
                                eo_df = pd.DataFrame({
                                    "Hráč": ownership_pct.index,
                                    "Vlastnictví v Top 10 (%)": ownership_pct.values
                                }).sort_values(by="Vlastnictví v Top 10 (%)", ascending=False)
                                
                                st.dataframe(
                                    eo_df.style.format({"Vlastnictví v Top 10 (%)": "{:.0f} %"})\
                                    .background_gradient(cmap='Reds', subset=['Vlastnictví v Top 10 (%)']),
                                    use_container_width=True, hide_index=True
                                )
                            else:
                                st.warning("Nepodařilo se načíst týmy rivalů (možná probíhá aktualizace kola).")
                    else:
                        st.error("Nepodařilo se načíst ligu. Zkontroluj, zda je ID správné a liga je veřejná.")
            else:
                st.warning("Nejprve si v levém panelu stáhni svůj tým, abychom ho mohli porovnat s rivaly!")
        else:
            st.warning("Zadej platné číselné ID ligy.")

with tab1:
    st.header("Matematický návrh přestupů (Jednorázový)")
    
    if len(my_team) == 15:
        if st.button("🚀 Spustit AI Optimalizaci", type="primary"):
            with st.spinner('AI prohledává miliony kombinací a skládá základní sestavu...'):
                current_squad_ids = df[df['unique_name'].isin(my_team)]['id'].tolist()
                current_squad_df = df[df['id'].isin(current_squad_ids)]
                total_budget = current_squad_df['now_cost'].sum() + bank
                
                current_squad_5gw_proj = current_squad_df['projected_5gw_fdr'].sum()

                prob = pulp.LpProblem("FPL_Transfer_Optimizer", pulp.LpMaximize)
                player_vars = pulp.LpVariable.dicts("player", df['id'], cat='Binary')
                
                if simulate_freehit:
                    projections = dict(zip(df['id'], df['projected_1gw_fdr']))
                else:
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

                if not (simulate_wildcard or simulate_freehit):
                    prob += pulp.lpSum([player_vars[i] for i in current_squad_ids]) >= (15 - free_transfers)

                prob.solve()
                
                if pulp.LpStatus[prob.status] == 'Optimal':
                    selected_ids = [i for i in df['id'] if player_vars[i].varValue == 1]
                    new_squad_df = df[df['id'].isin(selected_ids)]
                    
                    new_squad_5gw_proj = new_squad_df['projected_5gw_fdr'].sum()
                    new_start, new_bench, cap_id, vc_id, new_xi_1gw_proj = get_best_xi(new_squad_df)
                    
                    players_out = current_squad_df[~current_squad_df['id'].isin(selected_ids)]
                    players_in = new_squad_df[~new_squad_df['id'].isin(current_squad_ids)]
                    
                    if simulate_wildcard:
                        st.success("🃏 WILDCARD AKTIVOVÁN: Tým byl kompletně přestavěn s výhledem na 5 kol!")
                    elif simulate_freehit:
                        st.success("🆓 FREE HIT AKTIVOVÁN: Tým byl optimalizován POUZE pro příští kolo!")
                    elif simulate_bb:
                        st.success("🚀 BENCH BOOST AKTIVOVÁN: Lavička se počítá do celkového skóre!")
                    
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
                                    
                                    health_icon = f" <span title='{row['news']}' style='cursor: help;'>🏥</span>" if row['health_multiplier'] < 1.0 else ""
                                        
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
                            
                            bg_color = "rgba(44, 186, 0, 0.15)" if simulate_bb else "rgba(255, 99, 71, 0.1)"
                            border_color = "rgba(44, 186, 0, 0.5)" if simulate_bb else "rgba(255, 99, 71, 0.3)"
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 8px; background-color: {bg_color}; border-radius: 10px; border: 1px dashed {border_color};">
                                <div style="font-weight: bold; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['web_name']}{health_icon}</div>
                                <div style="font-size: 12px; color: gray;">{row['position']} | {row['projected_1gw_fdr']:.1f} b.</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    if simulate_bb:
                        captain_bonus = new_start.loc[new_start['id'] == cap_id, 'projected_1gw_fdr'].values[0]
                        expected_pts_display = new_squad_df['projected_1gw_fdr'].sum() + captain_bonus
                        pts_label = "Očekávané body (Všech 15 hráčů!)"
                    else:
                        expected_pts_display = new_xi_1gw_proj
                        pts_label = "Očekávané body sestavy (Příští kolo)"
                        
                    col_met1, col_met2, col_met3 = st.columns(3)
                    col_met1.metric("Zisk z přestupu (Celý tým na 5 kol)", f"+{new_squad_5gw_proj - current_squad_5gw_proj:.1f} bodů")
                    col_met2.metric(pts_label, f"{expected_pts_display:.1f} bodů")
                    col_met3.metric("Zůstatek v bance", f"{total_budget - new_squad_df['now_cost'].sum():.1f} m")
                else:
                    st.error("Nepodařilo se najít řešení. Zkontroluj rozpočet.")
    else:
        st.info(f"👈 Vyber v levém panelu přesně 15 hráčů. Zatím jich máš {len(my_team)}.")

with tab5:
    st.header("🚀 Vícekolový plánovač přestupů (Multi-Period)")
    st.write("Tento model chápe čas. Plánuje sekvenci přestupů na několik kol dopředu a matematicky kalkuluje, zda se vyplatí vzít hit (-4 body) pro zisk lepšího hráče.")
    
    horizon = st.slider("Plánovací horizont (počet kol dopředu):", min_value=2, max_value=5, value=3)
    
    if len(my_team) == 15:
        if st.button("🔮 Spustit Vícekolovou Optimalizaci", type="primary"):
            with st.spinner(f'AI simuluje všechny časové osy a cesty pro dalších {horizon} kol...'):
                current_squad_ids = df[df['unique_name'].isin(my_team)]['id'].tolist()
                total_budget = df[df['id'].isin(current_squad_ids)]['now_cost'].sum() + bank
                
                prob = pulp.LpProblem("MultiPeriod_FPL", pulp.LpMaximize)
                gw_range = range(1, horizon + 1)
                
                squad = pulp.LpVariable.dicts("squad", (df['id'], range(horizon + 1)), cat='Binary')
                transfer_in = pulp.LpVariable.dicts("transfer_in", (df['id'], gw_range), cat='Binary')
                transfer_out = pulp.LpVariable.dicts("transfer_out", (df['id'], gw_range), cat='Binary')
                
                transfers_count = pulp.LpVariable.dicts("transfers_count", gw_range, lowBound=0, cat='Integer')
                hits = pulp.LpVariable.dicts("hits", gw_range, lowBound=0, cat='Integer')
                
                costs = dict(zip(df['id'], df['now_cost']))
                names = dict(zip(df['id'], df['unique_name']))
                projections = {w: dict(zip(df['id'], df[f'proj_gw{w}'])) for w in gw_range}
                
                objective = []
                for w in gw_range:
                    for i in df['id']:
                        objective.append(squad[i][w] * projections[w][i])
                    objective.append(-4.0 * hits[w])
                prob += pulp.lpSum(objective)
                
                for i in df['id']:
                    if i in current_squad_ids:
                        prob += squad[i][0] == 1
                    else:
                        prob += squad[i][0] == 0
                        
                for w in gw_range:
                    prob += pulp.lpSum([squad[i][w] for i in df['id']]) == 15
                    prob += pulp.lpSum([squad[i][w] * costs[i] for i in df['id']]) <= total_budget
                    
                    prob += pulp.lpSum([squad[i][w] for i in df[df['position'] == 'GK']['id']]) == 2
                    prob += pulp.lpSum([squad[i][w] for i in df[df['position'] == 'DEF']['id']]) == 5
                    prob += pulp.lpSum([squad[i][w] for i in df[df['position'] == 'MID']['id']]) == 5
                    prob += pulp.lpSum([squad[i][w] for i in df[df['position'] == 'FWD']['id']]) == 3

                    for team in df['team_name'].unique():
                        prob += pulp.lpSum([squad[i][w] for i in df[df['team_name'] == team]['id']]) <= 3
                        
                    for i in df['id']:
                        prob += squad[i][w] == squad[i][w-1] + transfer_in[i][w] - transfer_out[i][w]
                        
                    prob += transfers_count[w] == pulp.lpSum([transfer_in[i][w] for i in df['id']])
                    
                    if w == 1:
                        prob += hits[w] >= transfers_count[w] - free_transfers
                    else:
                        prob += hits[w] >= transfers_count[w] - 1
                        
                prob.solve()
                
                if pulp.LpStatus[prob.status] == 'Optimal':
                    st.success(f"✅ Vícekolový plán na {horizon} kol úspěšně nalezen!")
                    
                    cols = st.columns(horizon)
                    for w, col in zip(gw_range, cols):
                        with col:
                            st.subheader(f"📅 GW +{w}")
                            
                            t_in = [names[i] for i in df['id'] if transfer_in[i][w].varValue == 1]
                            t_out = [names[i] for i in df['id'] if transfer_out[i][w].varValue == 1]
                            hit_cost = hits[w].varValue * 4
                            
                            st.markdown(f"**Očekávané body týmu:** {sum([projections[w][i] for i in df['id'] if squad[i][w].varValue == 1]):.1f}")
                            
                            if not t_in and not t_out:
                                st.info("🔄 Rolování přestupu (Žádná akce)")
                            else:
                                for p in t_out: st.error(f"❌ OUT: {p}")
                                for p in t_in: st.success(f"✅ IN: {p}")
                                
                                if hit_cost > 0:
                                    st.warning(f"⚠️ Hit: -{int(hit_cost)} bodů")
                                else:
                                    st.caption("Zdarma (v rámci FT)")
                else:
                    st.error("Nepodařilo se najít řešení. Zkontroluj rozpočet.")
    else:
        st.info(f"👈 Vyber v levém panelu přesně 15 hráčů.")

with tab6:
    st.header("©️ Pokročilý Plánovač Kapitánů")
    st.write("Rozhodování o kapitánovi vyhrává mini-ligy. Tento model analyzuje tvůj aktuální tým a rozkládá projekci na **Floor** (jistota bodů) a **Ceiling** (maximální potenciál).")

    if len(my_team) > 0:
        team_df = df[df['unique_name'].isin(my_team)].copy()

        team_df['Floor'] = 2.0 * team_df['health_multiplier']
        team_df.loc[team_df['position'].isin(['DEF', 'GK']), 'Floor'] += (team_df['CS_Prob'] / 100.0 * 4.0) * team_df['health_multiplier']
        team_df.loc[team_df['position'] == 'MID', 'Floor
'] += (team_df['CS_Prob'] / 100.0 * 1.0) * team_df['health_multiplier']

        team_df['Ceiling'] = team_df['projected_1gw_fdr'] + (team_df['Goal_Prob'] / 100.0 * 5.0) + (team_df['xA_90'] * 0.5) + 1.5

        top_caps = team_df.sort_values(by='projected_1gw_fdr', ascending=False).head(5)

        if len(top_caps) >= 3:
            c1, c2, c3 = st.columns(3)
            cap = top_caps.iloc[0]
            vc = top_caps.iloc[1]
            diff = top_caps.iloc[2]

            with c1:
                st.success(f"👑 KAPITÁN (C): {cap['web_name']}")
                st.markdown(f"**Zápas:** {cap['Zápas 1']} | **Projekce:** {cap['projected_1gw_fdr']:.1f} b.")
                st.progress(min(cap['Goal_Prob']/100.0, 1.0), text=f"Šance na gól: {cap['Goal_Prob']:.1f}%")

            with c2:
                st.info(f"🥈 ZÁSTUPCE (VC): {vc['web_name']}")
                st.markdown(f"**Zápas:** {vc['Zápas 1']} | **Projekce:** {vc['projected_1gw_fdr']:.1f} b.")
                st.progress(min(vc['Goal_Prob']/100.0, 1.0), text=f"Šance na gól: {vc['Goal_Prob']:.1f}%")

            with c3:
                st.warning(f"🎲 DIFERENCIÁL: {diff['web_name']}")
                st.markdown(f"**Zápas:** {diff['Zápas 1']} | **Projekce:** {diff['projected_1gw_fdr']:.1f} b.")
                st.progress(min(diff['Goal_Prob']/100.0, 1.0), text=f"Šance na gól: {diff['Goal_Prob']:.1f}%")

        st.divider()
        st.subheader("📊 Analýza rizika (Floor vs. Ceiling)")

        fig_cap = go.Figure()
        fig_cap.add_trace(go.Bar(x=top_caps['web_name'], y=top_caps['Floor'], name='Floor (Jistota)', marker_color='#2ca02c'))
        fig_cap.add_trace(go.Bar(x=top_caps['web_name'], y=top_caps['projected_1gw_fdr'] - top_caps['Floor'], name='Očekávané body', marker_color='#1f77b4'))
        fig_cap.add_trace(go.Bar(x=top_caps['web_name'], y=top_caps['Ceiling'] - top_caps['projected_1gw_fdr'], name='Ceiling (Potenciál)', marker_color='#ff7f0e'))

        fig_cap.update_layout(barmode='stack', title="Top 5 kandidátů na kapitána ve tvém týmu", xaxis_title="Hráč", yaxis_title="Body")
        st.plotly_chart(fig_cap, use_container_width=True)

    else:
        st.info("👈 Nejprve si v levém panelu stáhni nebo vyber svůj tým.")

with tab2:
    st.header("Kompletní databáze hráčů a Sázkové kurzy")
    
    all_teams = sorted(df['team_name'].unique().tolist())
    selected_teams = st.multiselect("🔍 Filtrovat podle klubů:", all_teams, default=[], placeholder="Vyber jeden nebo více týmů...")
    
    if selected_teams:
        filtered_df = df[df['team_name'].isin(selected_teams)]
    else:
        filtered_df = df
    
    display_df = filtered_df[['unique_name', 'position', 'now_cost', 'price_trend', 'net_transfers', 'odds_goal', 'odds_cs', 'projected_1gw_fdr', 'projected_5gw_fdr', 'Zápas 1', 'Zápas 2', 'Zápas 3', 'Zápas 4', 'Zápas 5']].copy()
    display_df.columns = ['Hráč (Tým)', 'Pozice', 'Cena', 'Cenový Trend', 'Čisté Přestupy', 'Kurz na Gól', 'Kurz na ČK', 'Hybridní Projekce (1 kolo)', 'Projekce (5 kol)', 'Zápas 1', 'Zápas 2', 'Zápas 3', 'Zápas 4', 'Zápas 5']
    
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
        'Cena': "{:.1f}", 'Čisté Přestupy': "{:,}", 'Kurz na Gól': "{:.2f}", 'Kurz na ČK': "{:.2f}", 'Hybridní Projekce (1 kolo)': "{:.1f}", 'Projekce (5 kol)': "{:.1f}"
    })
    
    st.dataframe(styled_df, use_container_width=True, height=600)

with tab3:
    st.header("🕸️ Porovnávač hráčů (Radarový graf)")
    st.write("Porovnej dva hráče vizuálně. Graf ukazuje **percentily** (0-100). Hodnota 90 znamená, že hráč je lepší než 90 % ligy v dané statistice.")

    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("Vyber 1. hráče:", all_player_names, index=0)
    with col2:
        p2_name = st.selectbox("Vyber 2. hráče:", all_player_names, index=1)

    if p1_name and p2_name:
        p1 = df[df['unique_name'] == p1_name].iloc[0]
        p2 = df[df['unique_name'] == p2_name].iloc[0]

        categories = ['Forma (xPts)', 'Projekce (1 kolo)', 'Šance na Gól', 'Šance na ČK', 'Oček. Asistence (xA/90)', 'Oček. Góly (xG/90)']

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[p1['Form_pct'], p1['Proj_pct'], p1['Goal_pct'], p1['CS_pct'], p1['xA_pct'], p1['xG_pct']],
            theta=categories,
            fill='toself',
            name=p1['web_name'],
            line_color='blue'
        ))

        fig.add_trace(go.Scatterpolar(
            r=[p2['Form_pct'], p2['Proj_pct'], p2['Goal_pct'], p2['CS_pct'], p2['xA_pct'], p2['xG_pct']],
            theta=categories,
            fill='toself',
            name=p2['web_name'],
            line_color='red'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Surová data")
        comp_df = pd.DataFrame({
            "Statistika": ["Cena", "Forma (xPts)", "Projekce (1 kolo)", "Šance na Gól", "Šance na Čisté konto", "xG na 90 min", "xA na 90 min"],
            p1['web_name']: [f"{p1['now_cost']}m", f"{p1['form']:.2f}", f"{p1['projected_1gw_fdr']:.2f}", f"{p1['Goal_Prob']:.1f} %", f"{p1['CS_Prob']:.1f} %", f"{p1['xG_90']:.2f}", f"{p1['xA_90']:.2f}"],
            p2['web_name']: [f"{p2['now_cost']}m", f"{p2['form']:.2f}", f"{p2['projected_1gw_fdr']:.2f}", f"{p2['Goal_Prob']:.1f} %", f"{p2['CS_Prob']:.1f} %", f"{p2['xG_90']:.2f}", f"{p2['xA_90']:.2f}"]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

with tab4:
    st.header("🧠 AI Analýza tiskových konferencí (Gemini 1.5 Flash)")
    st.write("Vlož text z tiskovky nebo novinky z Twitteru. Skutečná AI od Googlu text přečte, pochopí kontext zranění a automaticky upraví projekce hráčů v celém systému!")
    
    api_key = ""
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("🔐 API klíč byl bezpečně načten z nastavení Streamlit Cloud (Secrets)!")
    else:
        api_key = st.text_input("🔑 Zadej svůj Google Gemini API klíč (zdarma na Google AI Studio):", type="password")
        st.info("💡 Tip pro nasazení: Přidej si klíč do nastavení aplikace ve Streamlit Cloud (Settings -> Secrets) jako `GEMINI_API_KEY = 'tvůj_klíč'`, abys ho nemusel zadávat ručně.")
    
    news_text = st.text_area("Text z tiskovky (např. přepis slov Pepa Guardioly):", height=200, placeholder="Např.: Haaland si poranil hamstring a o víkendu nenastoupí. Foden je unavený, možná začne na lavičce...")
    
    if st.button("🧠 Analyzovat text a upravit projekce", type="primary"):
        if not api_key:
            st.error("⚠️ Musíš zadat API klíč! Získáš ho zdarma na https://aistudio.google.com/")
        elif not news_text:
            st.warning("⚠️ Nejprve vlož nějaký text k analýze.")
        else:
            with st.spinner("AI čte text a hledá zranění..."):
                try:
                    genai.configure(api_key=api_key)

                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    Jsi expert na Fantasy Premier League (FPL). Přečti si následující text z tiskové konference nebo zpráv a extrahuj informace o zraněních, rotaci nebo dostupnosti hráčů.
                    
                    Vrať POUZE validní JSON v tomto přesném formátu, nic jiného (žádný markdown, žádné vysvětlování):
                    {{
                        "players": [
                            {{"web_name": "Jméno hráče", "xMins_multiplier": 0.0, "reason": "Stručný důvod v češtině"}}
                        ]
                    }}
                    
                    Pravidla pro xMins_multiplier:
                    - 0.0 = Hráč je zraněný nebo suspendovaný a určitě nehraje.
                    - 0.5 = Hráč je nejistý (doubtful), má drobný šrám, nebo pravděpodobně začne na lavičce.
                    - 1.0 = Hráč je plně fit a připraven hrát.
                    
                    Text k analýze:
                    {news_text}
                    """
                    
                    response = model.generate_content(prompt)
                    cleaned_response = response.text.replace('```json', '').replace('```', '').strip()
                    extracted_data = json.loads(cleaned_response)['players']
                    
                    if extracted_data:
                        st.session_state['nlp_modifiers'] = extracted_data
                        st.success("✅ Analýza dokončena! Projekce hráčů byly upraveny.")
                        st.rerun()
                    else:
                        st.info("AI v textu nenašla žádné relevantní informace o zraněních.")
                        
                except Exception as e:
                    st.error(f"❌ Chyba při komunikaci s AI nebo při zpracování dat: {e}")
