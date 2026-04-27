import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pulp
import pandas as pd

# ZDE UPRAV NÁZEV SOUBORU, pokud se tvá hlavní aplikace nejmenuje app.py
from app import load_fpl_data, get_current_gw, fetch_manager_team, get_best_xi

def send_fpl_email(subject, body):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("EMAIL_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("✅ E-mail úspěšně odeslán!")
    except Exception as e:
        print(f"❌ Chyba při odesílání e-mailu: {e}")

def main():
    print("🤖 Spouštím FPL AI Bota...")
    
    # Načtení ID z GitHub Secrets (nebo z lokálního prostředí)
    manager_id = os.environ.get("FPL_MANAGER_ID")
    if not manager_id:
        print("⚠️ Chybí FPL_MANAGER_ID!")
        return

    # --- 1. NAČTENÍ DAT Z TVÉ APLIKACE ---
    print("📥 Stahuji data z FPL a počítám EMA/Poisson modely...")
    df = load_fpl_data()
    gw = get_current_gw()
    my_team, bank, real_gw = fetch_manager_team(manager_id, gw, df)
    
    if not my_team:
        print("❌ Nepodařilo se načíst tým.")
        return

    current_squad_ids = df[df['unique_name'].isin(my_team)]['id'].tolist()
    current_squad_df = df[df['id'].isin(current_squad_ids)]
    total_budget = current_squad_df['now_cost'].sum() + bank

    # --- 2. AI OPTIMALIZACE PŘESTUPŮ (PuLP) ---
    print("🧠 Spouštím matematickou optimalizaci (1 volný přestup)...")
    prob = pulp.LpProblem("FPL_Bot_Optimizer", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("player", df['id'], cat='Binary')
    
    projections = dict(zip(df['id'], df['projected_5gw_fdr']))
    costs = dict(zip(df['id'], df['now_cost']))
    
    # Cílová funkce: Maximalizovat body na 5 kol
    prob += pulp.lpSum([projections[i] * player_vars[i] for i in df['id']])
    
    # Omezení (15 hráčů, rozpočet, pozice, max 3 z týmu)
    prob += pulp.lpSum([player_vars[i] for i in df['id']]) == 15
    prob += pulp.lpSum([costs[i] * player_vars[i] for i in df['id']]) <= total_budget
    prob += pulp.lpSum([player_vars[i] for i in df[df['position'] == 'GK']['id']]) == 2
    prob += pulp.lpSum([player_vars[i] for i in df[df['position'] == 'DEF']['id']]) == 5
    prob += pulp.lpSum([player_vars[i] for i in df[df['position'] == 'MID']['id']]) == 5
    prob += pulp.lpSum([player_vars[i] for i in df[df['position'] == 'FWD']['id']]) == 3
    
    for team in df['team_name'].unique():
        prob += pulp.lpSum([player_vars[i] for i in df[df['team_name'] == team]['id']]) <= 3

    # Omezení na 1 volný přestup (musí zůstat alespoň 14 původních hráčů)
    prob += pulp.lpSum([player_vars[i] for i in current_squad_ids]) >= 14

    prob.solve()

    # --- 3. ZPRACOVÁNÍ VÝSLEDKŮ ---
    selected_ids = [i for i in df['id'] if player_vars[i].varValue == 1]
    new_squad_df = df[df['id'].isin(selected_ids)]
    
    players_out = current_squad_df[~current_squad_df['id'].isin(selected_ids)]
    players_in = new_squad_df[~new_squad_df['id'].isin(current_squad_ids)]
    
    # Výběr nejlepší sestavy a kapitána pro nové kolo
    new_start, new_bench, cap_id, vc_id, new_xi_1gw_proj = get_best_xi(new_squad_df)
    
    cap_row = new_start[new_start['id'] == cap_id].iloc[0]
    vc_row = new_start[new_start['id'] == vc_id].iloc[0]

    # Formátování textu přestupů
    transfer_text = ""
    if players_out.empty:
        transfer_text = "🔄 Žádný přestup není nutný (šetři volný přestup na další kolo).\n"
    else:
        for _, p_out in players_out.iterrows():
            transfer_text += f"❌ OUT: {p_out['unique_name']} ({p_out['now_cost']}m) | Projekce: {p_out['projected_5gw_fdr']:.1f} b.\n"
        for _, p_in in players_in.iterrows():
            transfer_text += f"✅ IN: {p_in['unique_name']} ({p_in['now_cost']}m) | Projekce: {p_in['projected_5gw_fdr']:.1f} b.\n"

    # --- 4. GENEROVÁNÍ TEXTU E-MAILU ---
    report_text = f"""Ahoj Manažere! 🤖

Tady je tvůj páteční FPL AI Report pro Gameweek {gw} (Tým ID: {manager_id}).

👑 DOPORUČENÝ KAPITÁN:
- (C) {cap_row['web_name']} (Zápas: {cap_row['Zápas 1']} | Projekce: {cap_row['projected_1gw_fdr']:.1f} b.)
- (VC) {vc_row['web_name']} (Zápas: {vc_row['Zápas 1']} | Projekce: {vc_row['projected_1gw_fdr']:.1f} b.)

🔄 NÁVRH PŘESTUPŮ (AI Optimalizace na 5 kol):
{transfer_text}
📊 OČEKÁVANÉ BODY SESTAVY: {new_xi_1gw_proj:.1f} bodů
💰 ZŮSTATEK V BANCE: {total_budget - new_squad_df['now_cost'].sum():.1f} m

Hodně štěstí o víkendu!
Tvůj Ultimátní FPL AI Manager
"""
    
    # --- 5. ODESLÁNÍ ---
    print("📧 Odesílám e-mail...")
    send_fpl_email(f"🚨 FPL AI Report: Gameweek {gw} je za rohem!", report_text)

if __name__ == "__main__":
    main()
