import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Zde naimportuj své funkce z tvého hlavního souboru (např. pokud se jmenuje app.py)
# from app import load_fpl_data, fetch_manager_team, get_best_xi, get_current_gw

def send_fpl_email(subject, body):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("EMAIL_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Použití utf-8 pro správné zobrazení české diakritiky a emoji
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        # Připojení na Gmail SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("✅ E-mail úspěšně odeslán!")
    except Exception as e:
        print(f"❌ Chyba při odesílání e-mailu: {e}")

def main():
    print("🤖 Spouštím FPL AI Bota...")
    manager_id = os.environ.get("FPL_MANAGER_ID")
    
    # --- 1. NAČTENÍ DAT (Odkomentuj a propoj se svou aplikací) ---
    # print("Stahuji data...")
    # df = load_fpl_data()
    # gw = get_current_gw()
    # my_team, bank, real_gw = fetch_manager_team(manager_id, gw, df)
    
    # --- 2. OPTIMALIZACE A VÝBĚR SESTAVY ---
    # current_squad_ids = df[df['unique_name'].isin(my_team)]['id'].tolist()
    # current_squad_df = df[df['id'].isin(current_squad_ids)]
    # start_df, bench_df, cap_id, vc_id, c_xi_pts = get_best_xi(current_squad_df)
    
    # cap_name = start_df[start_df['id'] == cap_id]['web_name'].values[0]
    # vc_name = start_df[start_df['id'] == vc_id]['web_name'].values[0]
    
    # --- 3. GENEROVÁNÍ TEXTU E-MAILU ---
    report_text = f"""Ahoj Manažere! 🤖
    
Tady je tvůj páteční FPL AI Report pro tvůj tým (ID: {manager_id}).

👑 DOPORUČENÝ KAPITÁN:
- (C) Erling Haaland (Ukázka - doplň proměnnou cap_name)
- (VC) Mohamed Salah (Ukázka - doplň proměnnou vc_name)

🔄 NÁVRH PŘESTUPŮ (PuLP Optimalizace):
❌ OUT: Bukayo Saka (Zranění)
✅ IN: Phil Foden (Skvělý los)
(Zde si napojíš výstup ze svého PuLP modelu)

Hodně štěstí o víkendu!
Tvůj Ultimátní FPL AI Manager
"""
    
    # --- 4. ODESLÁNÍ ---
    send_fpl_email("🚨 Tvůj páteční FPL AI Report je připraven!", report_text)

if __name__ == "__main__":
    main()
