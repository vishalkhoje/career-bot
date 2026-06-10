import sys
import os
import sqlite3
import json
import pandas as pd
import gradio as gr
import plotly.express as px
from datetime import datetime

# Path configuration
# Add the current directory to sys.path to ensure 'src' is found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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
        df = df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)
    return df

def load_benchmark_data():
    """Load historical benchmark results."""
    if not os.path.exists(EVAL_DB):
        return pd.DataFrame()
    
    conn = sqlite3.connect(EVAL_DB)
    try:
        df = pd.read_sql_query("SELECT * FROM benchmarks ORDER BY timestamp DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_dashboard_data(num_queries=10):
    """Aggregate data for the dashboard components, filtered by num_queries."""
    # Ensure database is up to date
    try:
        from src import EvaluationSystem
        EvaluationSystem(db_path=EVAL_DB)
    except ImportError:
        pass
    
    eval_df = load_eval_data()
    mon_df = load_monitoring_data()
    bench_df = load_benchmark_data()
    
    if eval_df.empty and mon_df.empty:
        return "No data found.", None, None, None, pd.DataFrame(), None
    
    # Ensure all expected columns exist
    expected_eval_cols = ['hallucination_score', 'relevance_score', 'correctness_score', 'groundedness_score', 'feedback']
    if not eval_df.empty:
        for col in expected_eval_cols:
            if col not in eval_df.columns:
                eval_df[col] = 0.0

    # ── DYNAMIC FILTERING ──────────────────────────────────────────────────
    # Slice the dataframes for the "Recent Performance" section
    recent_eval_df = eval_df.head(num_queries).copy()
    
    # Calculate KPIs for the selected range
    avg_latency = mon_df.head(num_queries)['latency_ms'].mean() if not mon_df.empty else 0
    total_cost = mon_df.head(num_queries)['cost_usd'].sum() if 'cost_usd' in mon_df else 0
    avg_correctness = recent_eval_df['correctness_score'].mean() if not recent_eval_df.empty else 0
    avg_relevance = recent_eval_df['relevance_score'].mean() if not recent_eval_df.empty else 0
    avg_hallucination = recent_eval_df['hallucination_score'].mean() if not recent_eval_df.empty else 0
    avg_groundedness = recent_eval_df['groundedness_score'].mean() if 'groundedness_score' in recent_eval_df else 0
    
    stats_markdown = f"""
    ### 📊 Key Performance Indicators (Last {num_queries} Queries)
    | Metric | Value |
    | :--- | :--- |
    | **Avg Latency** | {avg_latency:.2f} ms |
    | **Total Cost** | ${total_cost:.6f} |
    | **Avg Correctness** | {avg_correctness:.2f} |
    | **Avg Relevance** | {avg_relevance:.2f} |
    | **Avg Groundedness** | {avg_groundedness:.2f} |
    | **Avg Hallucination** | {avg_hallucination:.2f} (lower is better) |
    """
    
    # 2. Latency Over Time Plot (Full Trend)
    fig_latency = None
    if not mon_df.empty:
        fig_latency = px.line(mon_df, x='timestamp', y='latency_ms', title='Full Latency Trend (ms)',
                             labels={'latency_ms': 'Latency (ms)', 'timestamp': 'Time'})
        fig_latency.update_layout(template="plotly_dark")
    
    # 3. Live Chart: Metrics for selected N queries
    fig_live = None
    if not recent_eval_df.empty:
        # Use query preview for X-axis
        recent_eval_df['short_query'] = recent_eval_df['query'].str[:30] + "..."
        score_cols = ['correctness_score', 'relevance_score', 'groundedness_score']
        melted_recent = recent_eval_df.melt(id_vars=['short_query', 'timestamp'], value_vars=score_cols, 
                                            var_name='Metric', value_name='Score')
        
        fig_live = px.bar(melted_recent, x='short_query', y='Score', color='Metric', barmode='group',
                          title=f'Live Performance: Last {num_queries} Queries',
                          labels={'short_query': 'Query', 'Score': 'Score (0-1)'})
        fig_live.update_layout(template="plotly_dark", xaxis_tickangle=-45)

    # 4. Detailed Table (Merged) - Optimized for UI performance
    if not eval_df.empty and not mon_df.empty:
        # Extract request_id from metadata if available to prevent Cartesian products
        if 'metadata' in eval_df.columns:
            def get_req_id(m):
                try:
                    return json.loads(m).get('request_id') if isinstance(m, str) else None
                except:
                    return None
            eval_df['request_id'] = eval_df['metadata'].apply(get_req_id)
        else:
            eval_df['request_id'] = None
            
        if 'request_id' in mon_df.columns:
            # Join accurately on request_id to avoid multiplying rows for duplicate queries
            merged_df = pd.merge(eval_df, mon_df[['request_id', 'steps', 'cost_usd']], 
                                 on='request_id', how='left')
        else:
            # Fallback if no request_ids found
            eval_df['query_preview_id'] = eval_df['query'].str[:100]
            merged_df = pd.merge(eval_df, mon_df[['query_preview', 'steps', 'cost_usd']], 
                                 left_on='query_preview_id', right_on='query_preview', how='left')
            merged_df = merged_df.drop_duplicates(subset=['id']) if 'id' in merged_df.columns else merged_df
        
        cols_to_show = ['timestamp', 'query', 'response', 'groundedness_score', 'steps', 'cost_usd']
        available_cols = [c for c in cols_to_show if c in merged_df.columns]
        table_df = merged_df[available_cols].head(num_queries).copy()
    else:
        available_cols = [c for c in ['timestamp', 'query', 'response', 'groundedness_score'] if c in eval_df.columns]
        table_df = eval_df[available_cols].head(num_queries).copy() if not eval_df.empty else pd.DataFrame()
    
    # ── UI PERFORMANCE OPTIMIZATION ─────────────────────────────────────────
    # Truncate long text columns to prevent Gradio from hanging
    if not table_df.empty:
        if 'response' in table_df.columns:
            table_df['response'] = table_df['response'].apply(lambda x: (str(x)[:200] + '...') if len(str(x)) > 200 else x)
        if 'query' in table_df.columns:
            table_df['query'] = table_df['query'].apply(lambda x: (str(x)[:100] + '...') if len(str(x)) > 100 else x)
        if 'steps' in table_df.columns:
            table_df['steps'] = table_df['steps'].apply(lambda x: str(x).replace('[', '').replace(']', '').replace("'", ""))
    
    # Full score distribution for the other chart
    fig_scores = None
    if not eval_df.empty:
        score_cols = [c for c in ['correctness_score', 'relevance_score', 'hallucination_score', 'groundedness_score'] if c in eval_df.columns]
        melted_df = eval_df.melt(id_vars=['timestamp'], value_vars=score_cols, var_name='Metric', value_name='Score')
        fig_scores = px.box(melted_df, x='Metric', y='Score', title='All-Time Score Distribution')
        fig_scores.update_layout(template="plotly_dark")

    return stats_markdown, fig_latency, fig_scores, table_df, bench_df, fig_live

def build_dashboard():
    def trigger_benchmark():
        try:
            from tests.run_offline_eval import run_eval
            run_eval()
        except Exception as e:
            print(f"[Dashboard] Benchmark execution failed: {e}")
        return load_benchmark_data()

    with gr.Blocks(title="Career Bot Analytics", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Career Bot Analytics Dashboard")
        gr.Markdown("Deep observability into LLM performance, benchmarking, and cost.")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Data", variant="primary")
            num_queries_slider = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Analyze Last N Queries")
        
        with gr.Tabs():
            with gr.Tab("📈 Performance Overview"):
                with gr.Row():
                    stats_output = gr.Markdown("Loading statistics...")
                
                with gr.Row():
                    with gr.Column():
                        live_performance_plot = gr.Plot(label="Live Performance Chart")
                    with gr.Column():
                        latency_plot = gr.Plot(label="System Latency")
                
                with gr.Row():
                    with gr.Column():
                        scores_plot = gr.Plot(label="Score Distribution (All-Time)")

                gr.Markdown("### 📜 Recent Query Details")
                eval_table = gr.Dataframe(interactive=False)

            with gr.Tab("🧪 Benchmark Results"):
                gr.Markdown("### 🏁 Offline Gold Dataset Performance")
                with gr.Row():
                    run_bench_btn = gr.Button("🧪 Run Offline Benchmark (takes ~45s)", variant="secondary")
                bench_table = gr.Dataframe(interactive=False)
                
        def refresh(n):
            return get_dashboard_data(n)
        
        # Update on slider change or refresh button click
        refresh_btn.click(
            refresh, 
            inputs=[num_queries_slider],
            outputs=[stats_output, latency_plot, scores_plot, eval_table, bench_table, live_performance_plot]
        )
        
        run_bench_btn.click(
            trigger_benchmark,
            outputs=[bench_table]
        )
        
        num_queries_slider.change(
            refresh, 
            inputs=[num_queries_slider],
            outputs=[stats_output, latency_plot, scores_plot, eval_table, bench_table, live_performance_plot]
        )
        
        # Load data on start
        demo.load(refresh, inputs=[num_queries_slider], outputs=[stats_output, latency_plot, scores_plot, eval_table, bench_table, live_performance_plot])
        
    return demo

if __name__ == "__main__":
    dashboard = build_dashboard()
    dashboard.launch(server_name="0.0.0.0", server_port=7861)
