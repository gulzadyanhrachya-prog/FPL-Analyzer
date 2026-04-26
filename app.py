import streamlit as st
import pandas as pd
import requests
import pulp

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="FPL AI Manager", page_icon="⚽", layout="wide")
st.title("🤖 Ultimátní FPL AI Manager (Verze 2.0)")

# --- INICIALIZACE PAMĚTI (Session State) ---
# Aby si aplikace pamatovala tvůj tým po kliknutí na tlačítko
if 'my_team' not in st.session_state:
    st.session_state['my_team'] = []
if 'bank' not in st.session_state:
    st.session_state['bank'] = 0.0

# --- 1. DATOVÁ ČÁST (S využitím Cache) ---
@st.cache_data(ttl=3600)
def get_current_gw():
    """Zjistí aktuální odehrané kolo (Gameweek)"""
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    res = requests.get(url).json()
    for event in res['events']:
        if event['is_current']:
            return event['id']
    # Pokud se zrovna nehraje, vezme to nejbližší další
    for event in res['events']:
        if event['is_next']:
            return event['id']
    return 1

@st.cache_data(ttl=3600)
def load_fpl_data():
    """Stáhne a připraví kompletní data o hráčích a losu"""
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url).json()
    
    df = pd.DataFrame(response['elements'])
    teams_df = pd.DataFrame(response['teams'])
    team_mapping = dict(zip(teams_df['id'], teams_df['name']))
    df['team_name'] = df['team'].map(team_mapping)
    
    df['now_cost'] = df['now_cost'] / 10.0
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    df['position'] = df['element_type'].map(position_map)
    df['form'] = df['form'].astype(float)
    
    # FDR (Los)
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
    
    # Filtrace
    df = df[df['minutes'] > 500].copy()
    return df

def fetch_manager_team(manager_id, current_gw, df):
    """Stáhne tým konkrétního manažera podle jeho ID"""
    url = f'https://fantasy.premierleague.com/api/entry/{manager_id}/event/{current_gw}/picks/'
    res = requests.get(url)
    if res.status_code != 200:
        return None, None
    
    data = res.json()
    # Získáme ID hráčů a převedeme je na jména
    player_ids = [pick['element'] for pick in data['picks']]
    team_names = df[df['id'].isin(player_ids)]['web_name'].tolist()
    
    # Získáme peníze v bance
    bank = data['entry_history']['bank'] / 10.0
    return team_names, bank

df = load_fpl_data()

# --- 2. BOČNÍ PANEL (Uživatelské vstupy) ---
st.sidebar.header("📥 Import týmu")
manager_id = st.sidebar.text_input("Zadej své FPL ID (např. 123456):", help="Své ID najdeš v URL adrese, když si na webu FPL otevřeš záložku 'Points'.")

if st.sidebar.button("⬇️ Stáhnout můj tým", type="primary"):
    if manager_id.isdigit():
        with st.spinner("Stahuji data z FPL..."):
            gw = get_current_gw()
            fetched_team, fetched_bank = fetch_manager_team(manager_id, gw, df)
            
            if fetched_team:
                st.session_state['my_team'] = fetched_team
                st.session_state['bank'] = fetched_bank
                st.sidebar.success("✅ Tým úspěšně načten!")
            else:
                st.sidebar.error("❌ Tým nenalezen. Zkontroluj ID.")
    else:
        st.sidebar.error("⚠️ ID musí obsahovat pouze čísla.")

st.sidebar.divider()

st.sidebar.header("⚙️ Tvoje nastavení")
# Hodnoty se nyní berou z paměti (Session State), takže se po importu samy přepíšou!
bank = st.sidebar.number_input("Peníze v bance (miliony):", min_value=0.0, max_value=100.0, value=float(st.session_state['bank']), step=0.1)
free_transfers = st.sidebar.slider("Počet volných přestupů:", 1, 5, 1)

st.sidebar.subheader("Tvůj aktuální tým")
all_player_names = sorted(df['web_name'].tolist())

# Ošetření, aby se do výběru dostala jen platná jména
valid_team = [name for name in st.session_state['my_team'] if name in all_player_names]

my_team = st.sidebar.multiselect("Vyber přesně 15 hráčů:", all_player_names, default=valid_team, max_selections=15)

# --- 3. HLAVNÍ OBSAH (Záložky) ---
tab1, tab2, tab3 = st.tabs(["🔄 Optimalizátor přestupů", "📊 Datové centrum", "🎙️ AI Analýza zranění"])

with tab1:
    st.header("Matematický návrh přestupů")
    st.write("Zadej svůj tým v levém panelu a nech umělou inteligenci najít nejlepší přestup.")
    
    if len(my_team) == 15:
        if st.button("🚀 Spustit AI Optimalizaci", type="primary"):
            with st.spinner('AI prohledává miliony kombinací...'):
                current_squad_ids = df[df['web_name'].isin(my_team)]['id'].tolist()
                current_squad_df = df[df['id'].isin(current_squad_ids)]
                total_budget = current_squad_df['now_cost'].sum() + bank
                current_projection = current_squad_df['projected_5gw_fdr'].sum()

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
                    new_projection = new_squad_df['projected_5gw_fdr'].sum()
                    
                    players_out = current_squad_df[~current_squad_df['id'].isin(selected_ids)]
                    players_in = new_squad_df[~new_squad_df['id'].isin(current_squad_ids)]
                    
                    st.subheader("🔄 Doporučené přestupy:")
                    col1, col2 = st.columns(2)
                    with col1:
                        for _, p_out in players_out.iterrows():
                            st.error(f"❌ PRODEJ: {p_out['web_name']} ({p_out['now_cost']}m) | Projekce: {p_out['projected_5gw_fdr']:.1f} b.")
                    with col2:
                        for _, p_in in players_in.iterrows():
                            st.success(f"✅ KUP: {p_in['web_name']} ({p_in['now_cost']}m) | Projekce: {p_in['projected_5gw_fdr']:.1f} b.")
                            
                    st.divider()
                    st.metric("Čistý zisk z přestupu (5 kol)", f"+{new_projection - current_projection:.1f} bodů")
                    st.write(f"**Zůstatek v bance:** {total_budget - new_squad_df['now_cost'].sum():.1f} milionů")
                else:
                    st.error("Nepodařilo se najít řešení. Zkontroluj rozpočet.")
    else:
        st.info(f"👈 Vyber v levém panelu přesně 15 hráčů. Zatím jich máš {len(my_team)}.")

with tab2:
    st.header("Kompletní databáze hráčů")
    st.write("Tabulku můžeš libovolně řadit kliknutím na názvy sloupců.")
    display_df = df[['web_name', 'team_name', 'position', 'now_cost', 'form', 'fdr_multiplier', 'projected_5gw_fdr']]
    display_df.columns = ['Hráč', 'Tým', 'Pozice', 'Cena', 'Forma', 'FDR Koeficient', 'Projekce (5 kol)']
    st.dataframe(display_df.sort_values(by='Projekce (5 kol)', ascending=False), use_container_width=True)

with tab3:
    st.header("Zpracování tiskových konferencí (Demo)")
    st.write("Vlož text z tiskovky. V této demo verzi je ukázka simulované odpovědi LLM.")
    news_text = st.text_area("Text z tiskovky:", height=150, placeholder="Vlož text sem...")
    if st.button("🧠 Analyzovat text pomocí LLM"):
        if news_text:
            st.info("AI analyzuje text...")
            st.error("❌ Haaland - Zraněný (Projekce snížena na 0.0)")
            st.warning("⚠️ Foden - Unavený (Projekce snížena o 75%)")
            st.success("✅ De Bruyne - Plně zdráv (Projekce beze změny)")
        else:
            st.warning("Nejprve vlož nějaký text.")
