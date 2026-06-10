import pandas as pd
import json

# Read eval_df
import sqlite3
conn = sqlite3.connect("evaluations.db")
eval_df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY timestamp DESC", conn)
conn.close()

# Read mon_df
data = []
with open("career_bot_metrics.jsonl", "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            if "query_preview" in entry:
                data.append(entry)
        except:
            pass
mon_df = pd.DataFrame(data)

# Simulate dashboard merge
eval_df['query_preview_id'] = eval_df['query'].str[:100]
merged_df = pd.merge(eval_df, mon_df[['query_preview', 'steps', 'cost_usd']], 
                     left_on='query_preview_id', right_on='query_preview', how='left')

print("Eval rows:", len(eval_df))
print("Mon rows:", len(mon_df))
print("Merged rows:", len(merged_df))
