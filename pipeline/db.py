import sqlite3
import pandas as pd
import os

DB_PATH = "data/results.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bias_type TEXT,
            template TEXT,
            group_name TEXT,
            prompt TEXT,
            response TEXT,
            sentiment_score REAL,
            toxicity_score REAL,
            stereotype_score REAL,
            judge_sentiment REAL,
            fairness_concern INTEGER,
            judge_explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_results(records):
    conn = sqlite3.connect(DB_PATH)
    for r in records:
        conn.execute("""INSERT INTO responses 
            (bias_type, template, group_name, prompt, response,
             sentiment_score, toxicity_score, stereotype_score,
             judge_sentiment, fairness_concern, judge_explanation)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (r["bias_type"], r["template"], r["group"],
             r["prompt"], r["response"],
             r.get("sentiment_score", 0), r.get("toxicity_score", 0),
             r.get("stereotype_score", 0), r.get("judge_sentiment", 0),
             int(r.get("fairness_concern", False)),
             r.get("judge_explanation", "")))
    conn.commit()
    conn.close()

def load_results():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM responses", conn)
    conn.close()
    return df