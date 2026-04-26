import streamlit as st
import pandas as pd
import requests
import pulp

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="FPL AI Manager", page_icon="⚽", layout="wide")
st.title("🤖 Ultimátní FPL AI Manager (Verze 2.0)")

# --- INICIALIZACE PAMĚTI (Session State) ---
if 'my_team' not in st.session_state:
    st.session_state['my_team'] = []
if 'bank' not in st.session_state:
    st.session_state['bank'] = 0.0

# --- 1. DATOVÁ ČÁST ---
@st.cache_data(ttl=3600)
def get_current_gw():
    """Zjistí aktuální nebo poslední odehrané kolo (Gameweek)"""
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
    df['team_name'] = df['team'].map(team_mapping)
    
    df['unique_name'] = df['web_name'] + " (" + df['team_name'] + ")"
    df['now_cost'] = df['now_cost'] / 10.0
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    df['position'] = df['element_type'].map(position_map)
    df['form'] = df['form'].astype(float)
    
    fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/?future=1'
    fixtures_data = requests.get(fixtures_url).json()
    team_fdr = {team_id: [] for team_id in teams_df['id']}
    for f in fixtures_data:
        if len(team_fdr[f['team_h']]) < 5: team_fdr[f['team_h']].append(f['team_h_difficulty'])
        if len(team_fdr[f['team_a']]) < 5: team_fdr[f['team_a']].append(f['team_a_difficulty'])
            
    team_multiplier = {}
    for team_id, fdrs in team_fdr.items():
        avg_fdr = sum(fdrs) / len(fdrs) if len(fdrs) > 0 else 3.0
        team_multiplier[team_id] = round(1.0 + (3.0 - avg_fdr) * 0.2, 2)
        
    df['fdr_multiplier'] = df['team'].map(team_multiplier)
    df['projected_5gw_fdr'] = (df['form'] * 5) * df['fdr_multiplier']
    
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

# NOVÁ FUNKCE: Výběr ideální základní jedenáctky a kapitána
def get_best_xi(squad_df):
    squad = squad_df.sort_values(by='projected_5gw_fdr', ascending=False)
    
    gk = squad[squad['position'] == 'GK']
    df = squad[squad['position'] == 'DEF']
    md = squad[squad['position'] == 'MID']
    fw = squad[squad['position'] == 'FWD']
    
    # 1. Povinný základ (1 GK, 3 DEF, 2 MID, 1 FWD)
    start_idx = [
        gk.index[0],
        df.index[0], df.index[1], df.index[2],
        md.index[0], md.index[1],
        fw.index[0]
    ]
    
    # 2. Doplnění zbylých 4 hráčů do pole podle nejvyšší projekce
    remaining_idx = [idx for idx in squad.index if idx not in start_idx and idx != gk.index[1]]
    remaining_players = squad.loc[remaining_idx].sort_values(by='projected_5gw_fdr', ascending=False)
    
    start_idx.extend(remaining_players.index[:4])
    bench_idx = [gk.index[1]] + list(remaining_players.index[4:])
    
    # 3. Určení kapitána (hráč s nejvyšší projekcí v základu)
    start_df = squad.loc[start_idx].sort_values(by='projected_5gw_fdr', ascending=False)
    captain_id = start_df.iloc[0]['id']
    vc_id = start_df.iloc[1]['id']
    
    # 4. Seřazení pro hezký výpis (GK -> DEF -> MID -> FWD)
    start_df['pos_order'] = start_df['position'].map({'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    start_df = start_df.sort_values(by=['pos_order', 'projected_5gw_fdr'], ascending=[True, False])
    
    bench_df = squad.loc[bench_idx]
    
    # 5. Výpočet reálných bodů (Základní sestava + bonus pro kapitána)
    captain_points = start_df.loc[start_df['id'] == captain_id, 'projected_5gw_fdr'].values[0]
    total_xi_points = start_df['projected_5gw_fdr'].sum() + captain_points
    
    return start_df, bench_df, captain_id, vc_id, total_xi_points

df = load_fpl_data()

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

# --- 3. HLAVNÍ OBSAH ---
tab1, tab2, tab3 = st.tabs(["🔄 Optimalizátor přestupů", "📊 Datové centrum", "🎙️ AI Analýza zranění"])

with tab1:
    st.header("Matematický návrh přestupů")
    
    if len(my_team) == 15:
        if st.button("🚀 Spustit AI Optimalizaci", type="primary"):
            with st.spinner('AI prohledává miliony kombinací a skládá základní sestavu...'):
                current_squad_ids = df[df['unique_name'].isin(my_team)]['id'].tolist()
                current_squad_df = df[df['id'].isin(current_squad_ids)]
                total_budget = current_squad_df['now_cost'].sum() + bank
                
                # Získáme reálnou projekci současného týmu (Základní XI + Kapitán)
                _, _, _, _, current_projection = get_best_xi(current_squad_df)

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
                    
                    # Získáme reálnou projekci a sestavu NOVÉHO týmu
                    new_start, new_bench, cap_id, vc_id, new_projection = get_best_xi(new_squad_df)
                    
                    players_out = current_squad_df[~current_squad_df['id'].isin(selected_ids)]
                    players_in = new_squad_df[~new_squad_df['id'].isin(current_squad_ids)]
                    
                    st.subheader("🔄 Doporučené přestupy:")
                    col1, col2 = st.columns(2)
                    with col1:
                        for _, p_out in players_out.iterrows():
                            st.error(f"❌ PRODEJ: {p_out['unique_name']} ({p_out['now_cost']}m)")
                    with col2:
                        for _, p_in in players_in.iterrows():
                            st.success(f"✅ KUP: {p_in['unique_name']} ({p_in['now_cost']}m)")
                            
                    st.divider()
                    
                    # VYKRESLENÍ ZÁKLADNÍ SESTAVY A KAPITÁNA
                    st.subheader("👑 Ideální základní sestava (Po přestupech)")
                    col_xi, col_bench = st.columns([2, 1])
                    
                    with col_xi:
                        st.markdown("**Základní jedenáctka:**")
                        for _, row in new_start.iterrows():
                            role = ""
                            if row['id'] == cap_id:
                                role = " **(C)** 🌟"
                            elif row['id'] == vc_id:
                                role = " *(VC)*"
                            st.write(f"`{row['position']}` | {row['unique_name']}{role} - {row['projected_5gw_fdr']:.1f} b.")
                            
                    with col_bench:
                        st.markdown("**🪑 Lavička (0 bodů):**")
                        for _, row in new_bench.iterrows():
                            st.write(f"`{row['position']}` | {row['unique_name']} - {row['projected_5gw_fdr']:.1f} b.")
                    
                    st.divider()
                    st.metric("Čistý zisk z přestupu (Reálné body za 5 kol)", f"+{new_projection - current_projection:.1f} bodů")
                    st.write(f"**Zůstatek v bance:** {total_budget - new_squad_df['now_cost'].sum():.1f} milionů")
                else:
                    st.error("Nepodařilo se najít řešení. Zkontroluj rozpočet.")
    else:
        st.info(f"👈 Vyber v levém panelu přesně 15 hráčů. Zatím jich máš {len(my_team)}.")

with tab2:
    st.header("Kompletní databáze hráčů")
    display_df = df[['unique_name', 'position', 'now_cost', 'form', 'fdr_multiplier', 'projected_5gw_fdr']]
    display_df.columns = ['Hráč (Tým)', 'Pozice', 'Cena', 'Forma', 'FDR Koeficient', 'Projekce (5 kol)']
    st.dataframe(display_df.sort_values(by='Projekce (5 kol)', ascending=False), use_container_width=True)

with tab3:
    st.header("Zpracování tiskových konferencí (Demo)")
    news_text = st.text_area("Text z tiskovky:", height=150, placeholder="Vlož text sem...")
    if st.button("🧠 Analyzovat text pomocí LLM"):
        if news_text:
            st.info("AI analyzuje text...")
            st.error("❌ Haaland - Zraněný (Projekce snížena na 0.0)")
            st.warning("⚠️ Foden - Unavený (Projekce snížena o 75%)")
            st.success("✅ De Bruyne - Plně zdráv (Projekce beze změny)")
        else:
            st.warning("Nejprve vlož nějaký text.")
