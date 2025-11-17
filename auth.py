import sqlite3
import hashlib

DB_PATH = "users.db"

def create_users_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT,
                    conversations TEXT
                )''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password, conversations) VALUES (?, ?, ?)",
              (username, hash_password(password), "[]"))
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[0] == hash_password(password):
        return True
    return False

def load_conversation(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT conversations FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else "[]"

def save_conversation(username, conversation_json):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET conversations = ? WHERE username = ?", (conversation_json, username))
    conn.commit()
    conn.close()
