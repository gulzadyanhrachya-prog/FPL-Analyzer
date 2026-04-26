import streamlit as st
import pandas as pd
import requests
import pulp
import json
from openai import OpenAI

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="FPL AI Manager", page_icon="⚽", layout="wide")
st.title("🤖 Ultimátní FPL AI Manager (Verze 2.0)")

# --- INICIALIZACE PAMĚTI (Session State) ---
if 'my_team' not in st.session_state:
    st.session_state['my_team'] = []
if 'bank' not in st.session_state:
    st.session_state['bank'] = 0.0
if 'nlp_modifiers' not in st.session_state:
    st.session_state['nlp_modifiers'] = []

# --- 1. DATOVÁ ČÁST ---
@st.cache_data(ttl=3600)
def get_current_gw():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    res = requests.get(url).json()
    for event in res['events']:
        if event['is_current']: return event['id']
    for event in res['events']:
        if event['is_previous']: return event['id']
    return 1

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
    df['form'] = df['form'].astype(float)
    
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
    
    df['projected_5gw_fdr'] = (df['form'] * 5) * df['fdr_multiplier_5gw']
    df['projected_1gw_fdr'] = (df['form'] * 1) * df['fdr_multiplier_1gw']
    
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
df = load_fpl_data()

# --- APLIKACE NLP MODIFIKÁTORŮ Z TISKOVEK ---
if st.session_state['nlp_modifiers']:
    for mod in st.session_state['nlp_modifiers']:
        idx = df['web_name'] == mod['web_name']
        df.loc[idx, 'projected_5gw_fdr'] *= mod['xMins_multiplier']
        df.loc[idx, 'projected_1gw_fdr'] *= mod['xMins_multiplier']

# --- 2. BOČNÍ PANEL ---
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

st.sidebar.header("⚙️ Tvoje nastavení")
bank = st.sidebar.number_input("Peníze v bance (miliony):", min_value=0.0, max_value=100.0, value=float(st.session_state['bank']), step=0.1)
free_transfers = st.sidebar.slider("Počet volných přestupů:", 1, 5, 1)

st.sidebar.subheader("Tvůj aktuální tým")
all_player_names = sorted(df['unique_name'].tolist())
valid_team = [name for name in st.session_state['my_team'] if name in all_player_names]
my_team = st.sidebar.multiselect("Vyber přesně 15 hráčů:", all_player_names, default=valid_team, max_selections=15)

if st.session_state['nlp_modifiers']:
    st.sidebar.divider()
    st.sidebar.subheader("🏥 Aktivní AI hlášení")
    for mod in st.session_state['nlp_modifiers']:
        if mod['xMins_multiplier'] < 1.0:
            st.sidebar.error(f"**{mod['web_name']}**: {mod['reason']}")

# --- 3. HLAVNÍ OBSAH ---
tab1, tab2, tab3 = st.tabs(["🔄 Optimalizátor přestupů", "📅 Databáze & Fixture Ticker", "🧠 AI Analýza tiskovek"])

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
                                        
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 10px; background-color: rgba(150, 150, 150, 0.1); border-radius: 10px; border: 1px solid rgba(150, 150, 150, 0.2); margin-bottom: 10px;">
                                        <div style="font-size: 28px;">👕</div>
                                        <div style="font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{row['web_name']}">{row['web_name']}{role}</div>
                                        <div style="font-size: 14px; color: #4CAF50; font-weight: bold;">{row['projected_1gw_fdr']:.1f} b.</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("🪑 Lavička náhradníků")
                    bench_cols = st.columns(4)
                    for col, (_, row) in zip(bench_cols, new_bench.iterrows()):
                        with col:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 8px; background-color: rgba(255, 99, 71, 0.1); border-radius: 10px; border: 1px dashed rgba(255, 99, 71, 0.3);">
                                <div style="font-weight: bold; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['web_name']}</div>
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
    st.header("Kompletní databáze hráčů a Fixture Ticker")
    
    display_df = df[['unique_name', 'position', 'now_cost', 'form', 'projected_1gw_fdr', 'projected_5gw_fdr', 'Zápas 1', 'Zápas 2', 'Zápas 3', 'Zápas 4', 'Zápas 5']].copy()
    display_df.columns = ['Hráč (Tým)', 'Pozice', 'Cena', 'Forma', 'Projekce (1 kolo)', 'Projekce (5 kol)', 'Zápas 1', 'Zápas 2', 'Zápas 3', 'Zápas 4', 'Zápas 5']
    
    diff_df = df[['Diff 1', 'Diff 2', 'Diff 3', 'Diff 4', 'Diff 5']].copy()
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
        'Cena': "{:.1f}", 'Forma': "{:.1f}", 'Projekce (1 kolo)': "{:.1f}", 'Projekce (5 kol)': "{:.1f}"
    })
    
    st.dataframe(styled_df, use_container_width=True, height=600)

with tab3:
    st.header("🧠 AI Analýza tiskových konferencí")
    st.write("Vlož text z tiskovky. AI z něj extrahuje zranění a automaticky upraví projekce hráčů v celém systému!")
    
    api_key = st.text_input("Zadej svůj OpenAI API klíč (začíná na sk-...):", type="password", help="Pokud klíč nemáš, nech pole prázdné a vyzkoušej si Demo režim.")
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
                        client = OpenAI(api_key=api_key)
                        prompt = """
                        Jsi expert na Fantasy Premier League. Přečti si text.
                        Vrať POUZE validní JSON ve formátu:
                        {"players": [{"web_name": "Jméno", "xMins_multiplier": 0.0, "reason": "Důvod"}]}
                        xMins_multiplier je 0.0 (nehraje), 0.25 (lavička), 0.75 (střídá), 1.0 (hraje).
                        """
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": news_text}
                            ],
                            response_format={"type": "json_object"}
                        )
                        extracted_data = json.loads(response.choices[0].message.content)['players']
                    
                    st.session_state['nlp_modifiers'] = extracted_data
                    st.success("✅ Analýza dokončena! Projekce hráčů byly upraveny.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Nastala chyba při komunikaci s AI: {e}")
