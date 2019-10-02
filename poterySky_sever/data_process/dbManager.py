from .config import config
import sqlite3

# conn = sqlite3.connect(config.db_path)
# db = conn.cursor()

def createDbConnect(type='r'):
    conn = sqlite3.connect(config.db_path)
    return conn.cursor()