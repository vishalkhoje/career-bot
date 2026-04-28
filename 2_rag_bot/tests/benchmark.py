import os
import sys
import json
from datetime import datetime
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import CareerAgent
from src.evaluation import EvaluationSystem
from src import config

def run_benchmark():
    print(f"\n{'='*60}")
    print(f"🚀 CAREER BOT BENCHMARK RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    agent = CareerAgent()
    eval_sys = EvaluationSystem()
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"[Benchmark] Loaded {len(dataset)} queries from gold dataset.\n")

    results = []
    
    for i, item in enumerate(dataset):
        query = item["query"]
        expected = item["expected_answer"]
        category = item["category"]
        
        print(f"[{i+1}/{len(dataset)}] Category: {category}")
        print(f"Query: {query}")
        
        # 1. Generate Response
        # We need to capture the context, but CareerAgent.chat returns a generator.
        # We'll use a wrapper or just collect the output.
        full_response = ""
        for chunk in agent.chat(query, []):
            full_response += chunk
        
        # 2. Get Context (Hack to get context used by agent)
        # In a real system, the agent would return context. 
        # Here we'll manually retrieve for the evaluation step to ensure it's fair.
        from src.retriever import CareerRetriever
        retriever = CareerRetriever()
        context = retriever.retrieve(query)

        # 3. Auto-Evaluate
        print("Grading...")
        scores = eval_sys.auto_evaluate(query, full_response, context)
        
        results.append({
            "query": query,
            "category": category,
            "response": full_response,
            "scores": scores
        })
        
        print(f"Scores -> Groundedness: {scores['groundedness']:.2f} | Relevance: {scores['relevance']:.2f} | Hallucination: {scores['hallucination']:.2f}\n")

    # 4. Aggregate Results
    avg_groundedness = statistics.mean([r["scores"]["groundedness"] for r in results])
    avg_relevance = statistics.mean([r["scores"]["relevance"] for r in results])
    avg_hallucination = statistics.mean([r["scores"]["hallucination"] for r in results])
    avg_correctness = statistics.mean([r["scores"]["correctness"] for r in results])

    summary = {
        "avg_groundedness": avg_groundedness,
        "avg_relevance": avg_relevance,
        "avg_hallucination": avg_hallucination,
        "avg_correctness": avg_correctness,
        "total_queries": len(results),
        "queries": results
    }

    # 5. Log to DB
    version = f"v{config.EMBEDDING_VERSION}"
    eval_sys.log_benchmark(version, summary)

    print(f"{'='*60}")
    print(f"📊 BENCHMARK COMPLETE (Version: {version})")
    print(f"Groundedness:  {avg_groundedness:.2f}")
    print(f"Relevance:     {avg_relevance:.2f}")
    print(f"Hallucination: {avg_hallucination:.2f} (lower is better)")
    print(f"Correctness:   {avg_correctness:.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_benchmark()
