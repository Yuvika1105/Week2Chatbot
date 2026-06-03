import json
import os
from datetime import datetime

def get_interpretation(metric_name: str, score: float) -> str:
    """Return the qualitative performance interpretation based on the metric score."""
    if metric_name in ["Context Precision", "Context Recall"]:
        if score >= 0.8: return "Excellent"
        if score >= 0.5: return "Acceptable"
        return "Investigate"
    elif metric_name == "Answer Faithfulness":
        if score >= 0.9: return "Excellent"
        if score >= 0.7: return "Acceptable"
        return "Investigate"
    elif metric_name == "Answer Relevance":
        if score >= 0.7: return "Excellent"
        if score >= 0.4: return "Acceptable"
        return "Investigate"
    elif metric_name == "Source Accuracy":
        if score >= 1.0: return "Excellent"
        if score >= 0.5: return "Acceptable"
        return "Investigate"
    elif metric_name == "Answer F1":
        if score >= 0.8: return "Excellent"
        if score >= 0.5: return "Partially correct"
        if score >= 0.3: return "Investigate"
        return "Poor"
    return ""

def save_report(results: dict, output_path: str) -> None:
    """Save results as JSON with a timestamp and print a human-readable summary table to stdout."""
    # 1. Add timestamp
    timestamp = datetime.now().isoformat()
    results["timestamp"] = timestamp
    
    # 2. Write JSON to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # 3. Print elegant summary table
    agg = results.get("aggregate", {})
    num_questions = len(results.get("per_question", []))
    
    # Extract values safely
    cp = agg.get("context_precision", 0.0)
    cr = agg.get("context_recall", 0.0)
    af = agg.get("faithfulness", 0.0)
    ar = agg.get("answer_relevance", 0.0)
    sa = agg.get("source_accuracy", 0.0)
    f1_dict = agg.get("answer_f1", {})
    f1_val = f1_dict.get("f1", 0.0)
    f1_prec = f1_dict.get("precision", 0.0)
    f1_rec = f1_dict.get("recall", 0.0)
    
    print("\nEvaluation Summary")
    print("==========================================================")
    print(" Metric                     Score    Interpretation")
    print("----------------------------------------------------------")
    print(f" Context Precision           {cp:.2f}    {get_interpretation('Context Precision', cp)}")
    print(f" Context Recall              {cr:.2f}    {get_interpretation('Context Recall', cr)}")
    print(f" Answer Faithfulness         {af:.2f}    {get_interpretation('Answer Faithfulness', af)}")
    print(f" Answer Relevance            {ar:.2f}    {get_interpretation('Answer Relevance', ar)}")
    print(f" Source Accuracy             {sa:.2f}    {get_interpretation('Source Accuracy', sa)}")
    print(f" Answer F1                   {f1_val:.2f}    {get_interpretation('Answer F1', f1_val)}")
    print(f"   +- Precision              {f1_prec:.2f}")
    print(f"   +- Recall                 {f1_rec:.2f}")
    print("==========================================================")
    print(f" Questions evaluated: {num_questions}")
    print(f" Report saved: {output_path}")
    print()
