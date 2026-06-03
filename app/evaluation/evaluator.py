from typing import Callable
import json
from app.evaluation.metrics import (
    context_precision,
    context_recall,
    answer_faithfulness,
    answer_relevance,
    source_accuracy,
    answer_f1
)

class RAGEvaluator:
    def __init__(self, rag_function: Callable, user_id: str = "u001"):
        self.rag_function = rag_function
        self.user_id = user_id

    def evaluate_single(self, entry: dict) -> dict:
        # Run one golden-dataset entry through rag_function and return all metrics.
        res = self.rag_function(entry["question"], user_id=self.user_id)
        
        generated_answer = res.get("response", "")
        predicted_sources = res.get("sources", [])
        retrieved_chunks = res.get("retrieved_chunks", [])
        
        relevant_chunks = entry.get("relevant_chunks", [])
        ground_truth_answer = entry.get("ground_truth_answer", "")
        ground_truth_sources = entry.get("ground_truth_sources", [])
        
        c_precision = context_precision(retrieved_chunks, relevant_chunks)
        c_recall = context_recall(retrieved_chunks, relevant_chunks)
        faithfulness = answer_faithfulness(generated_answer, retrieved_chunks)
        relevance = answer_relevance(entry["question"], generated_answer)
        s_accuracy = source_accuracy(predicted_sources, ground_truth_sources)
        f1_dict = answer_f1(generated_answer, ground_truth_answer)
        
        return {
            "question": entry["question"],
            "context_precision": c_precision,
            "context_recall": c_recall,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "source_accuracy": s_accuracy,
            "answer_f1": f1_dict
        }

    def evaluate(self, eval_set_path: str) -> dict:
        # Loads golden dataset, runs rag_function on every question,
        # computes all six metrics, returns aggregate + per-question results.
        with open(eval_set_path, "r", encoding="utf-8") as f:
            eval_set = json.load(f)
            
        per_question = []
        for entry in eval_set:
            metrics = self.evaluate_single(entry)
            per_question.append(metrics)
            
        # Aggregate metrics
        num_questions = len(per_question)
        if num_questions == 0:
            return {
                "aggregate": {
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevance": 0.0,
                    "source_accuracy": 0.0,
                    "answer_f1": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                },
                "per_question": []
            }
            
        agg_c_precision = sum(q["context_precision"] for q in per_question) / num_questions
        agg_c_recall = sum(q["context_recall"] for q in per_question) / num_questions
        agg_faithfulness = sum(q["faithfulness"] for q in per_question) / num_questions
        agg_relevance = sum(q["answer_relevance"] for q in per_question) / num_questions
        agg_source_accuracy = sum(q["source_accuracy"] for q in per_question) / num_questions
        
        agg_precision = sum(q["answer_f1"]["precision"] for q in per_question) / num_questions
        agg_recall = sum(q["answer_f1"]["recall"] for q in per_question) / num_questions
        agg_f1 = sum(q["answer_f1"]["f1"] for q in per_question) / num_questions
        
        return {
            "aggregate": {
                "context_precision": agg_c_precision,
                "context_recall": agg_c_recall,
                "faithfulness": agg_faithfulness,
                "answer_relevance": agg_relevance,
                "source_accuracy": agg_source_accuracy,
                "answer_f1": {
                    "precision": agg_precision,
                    "recall": agg_recall,
                    "f1": agg_f1
                }
            },
            "per_question": per_question
        }
