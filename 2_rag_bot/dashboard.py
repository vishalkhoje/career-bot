import sqlite3
import json
import pandas as pd
import gradio as gr
import plotly.express as px
import os
from datetime import datetime

# Path configuration
EVAL_DB = "evaluations.db"
METRICS_FILE = "career_bot_metrics.jsonl"

def load_eval_data():
    """Load evaluation metrics from SQLite."""
    if not os.path.exists(EVAL_DB):
        return pd.DataFrame()
    
    conn = sqlite3.connect(EVAL_DB)
    df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY timestamp DESC", conn)
    conn.close()
    
    # Convert timestamp to datetime
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def load_monitoring_data():
    """Load system metrics from JSONL."""
    if not os.path.exists(METRICS_FILE):
        return pd.DataFrame()
    
    data = []
    with open(METRICS_FILE, "r") as f:
        for line in f:
            try:
                # Filter out tool execution events if needed, or handle them
                entry = json.loads(line)
                if "query_preview" in entry: # Chat request event
                    data.append(entry)
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_dashboard_data():
    """Aggregate data for the dashboard components."""
    # Ensure database is up to date with new columns (like correctness_score)
    try:
        from src.evaluation import EvaluationSystem
        EvaluationSystem(db_path=EVAL_DB) # This triggers _init_db() and migration
    except ImportError:
        print("[Dashboard] Warning: Could not import EvaluationSystem for migration.")
    
    eval_df = load_eval_data()
    mon_df = load_monitoring_data()
    
    if eval_df.empty and mon_df.empty:
        return "No data found.", None, None, None, None, None
    
    # Ensure all expected columns exist to avoid KeyError
    expected_eval_cols = ['hallucination_score', 'relevance_score', 'correctness_score', 'feedback']
    if not eval_df.empty:
        for col in expected_eval_cols:
            if col not in eval_df.columns:
                eval_df[col] = 0.0

    # 1. Summary Metrics
    avg_latency = mon_df['latency_ms'].mean() if not mon_df.empty else 0
    avg_correctness = eval_df['correctness_score'].mean() if not eval_df.empty else 0
    avg_relevance = eval_df['relevance_score'].mean() if not eval_df.empty else 0
    avg_hallucination = eval_df['hallucination_score'].mean() if not eval_df.empty else 0
    
    stats_markdown = f"""
    ### 📊 Key Performance Indicators
    | Metric | Average Value |
    | :--- | :--- |
    | **Latency** | {avg_latency:.2f} ms |
    | **Correctness** | {avg_correctness:.2f} |
    | **Relevance** | {avg_relevance:.2f} |
    | **Hallucination** | {avg_hallucination:.2f} |
    """
    
    # 2. Latency Over Time Plot
    fig_latency = None
    if not mon_df.empty:
        fig_latency = px.line(mon_df, x='timestamp', y='latency_ms', title='Latency Trend (ms)',
                             labels={'latency_ms': 'Latency (ms)', 'timestamp': 'Time'})
        fig_latency.update_layout(template="plotly_dark")
    
    # 3. Scores Distribution
    fig_scores = None
    if not eval_df.empty:
        score_cols = [c for c in ['correctness_score', 'relevance_score', 'hallucination_score'] if c in eval_df.columns]
        if score_cols:
            melted_df = eval_df.melt(id_vars=['timestamp'], value_vars=score_cols, 
                                     var_name='Metric', value_name='Score')
            fig_scores = px.box(melted_df, x='Metric', y='Score', title='Evaluation Score Distribution')
            fig_scores.update_layout(template="plotly_dark")

    # 4. Recent Evaluations Table
    available_cols = [c for c in ['timestamp', 'query', 'response', 'correctness_score', 'relevance_score', 'feedback'] if c in eval_df.columns]
    table_df = eval_df[available_cols].head(10) if not eval_df.empty else pd.DataFrame()
    
    return stats_markdown, fig_latency, fig_scores, table_df

def build_dashboard():
    with gr.Blocks(title="Career Bot Analytics", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Career Bot Analytics Dashboard")
        gr.Markdown("Real-time monitoring of LLM evaluations and system performance.")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Data", variant="primary")
        
        with gr.Row():
            stats_output = gr.Markdown("Loading statistics...")
            
        with gr.Row():
            with gr.Column():
                latency_plot = gr.Plot(label="System Latency")
            with gr.Column():
                scores_plot = gr.Plot(label="LLM Evaluation Scores")
                
        gr.Markdown("### 📜 Recent Evaluations")
        eval_table = gr.Dataframe(interactive=False)
        
        def refresh():
            return get_dashboard_data()
        
        refresh_btn.click(
            refresh, 
            outputs=[stats_output, latency_plot, scores_plot, eval_table]
        )
        
        # Load data on start
        demo.load(refresh, outputs=[stats_output, latency_plot, scores_plot, eval_table])
        
    return demo

if __name__ == "__main__":
    dashboard = build_dashboard()
    dashboard.launch(server_name="0.0.0.0", server_port=7861)
