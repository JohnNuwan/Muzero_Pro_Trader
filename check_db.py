import sqlite3

conn = sqlite3.connect('gemini_v19/live/trades.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM trades')
count = cursor.fetchone()[0]
print(f'✅ Total experiences collected: {count}')

if count > 0:
    cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM trades')
    first, last = cursor.fetchone()
    print(f'📅 First trade: {first}')
    print(f'📅 Last trade: {last}')
    
    cursor.execute('SELECT symbol, COUNT(*) as cnt FROM trades GROUP BY symbol ORDER BY cnt DESC')
    print('\n📊 Trades per symbol:')
    for row in cursor.fetchall():
        print(f'  {row[0]}: {row[1]} experiences')

conn.close()
