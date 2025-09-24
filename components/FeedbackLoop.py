"""
The feedback loop can be considered as the training phase for improving the rubric criteria
for classifying the intent rating of conversations between a client and an agent.
"""
from config import (
    MAX_CONSECUTIVE_NO_IMPROVEMENT,
    IDENTIFY_RUBRIC_ISSUES_PROMPT,
    CONSECUTIVE_NO_IMPROVEMENT,
    SUMMARIZE_RUBRIC_PROMPT,
    MODIFY_RUBRIC_PROMPT,
    MAX_CONCURRENT,
    TEMPERATURE,
    MAX_ITER
)

from components import ConversationExtractor, ConversationPipeline, RubricProcessor
from clients import BigQueryClient, OpenAIClient

from datetime import datetime
import pandas as pd
import logging
import asyncio
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FeedbackLoop:
    def __init__(
        self,
        openai: OpenAIClient,
        bq: BigQueryClient,
        limit: int,
        rubric_prompt: str,
        rubric_intent: str,
        iterations: int = MAX_ITER,
        model: str = "gpt-4.1-mini",
        temperature: float = TEMPERATURE
    ):
        self.openai = openai
        self.bq = bq
        self.limit = limit
        self.rubric_prompt = rubric_prompt
        self.rubric_intent = rubric_intent
        self.iterations = iterations
        self.model = model
        self.temperature = temperature
        self.rubric_history = []
        self._conversation_cache = {}
        self.accuracies = []

    @staticmethod
    def transform_scorecard_results(results: dict, ticket_id: str = None):
        records = []

        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            for ticket, details in results.items():
                if "result" in details and isinstance(details["result"], dict):
                    records.append({
                        "ticket_id": ticket,
                        "top_intent": details["result"]["top_intent"]
                    })
        elif ticket_id is not None and isinstance(results, dict):
            if "result" in results and isinstance(results["result"], dict):
                records.append({
                    "ticket_id": ticket_id,
                    "top_intent": results["result"]["top_intent"]
                })
        else:
            raise ValueError("Unsupported format. Check the results again.")
        return records

    @staticmethod
    def get_run_dir(base_dir: dir = "rubrics"):
        ts = datetime.now().strftime("%Y-%m-%d")
        run_dir = os.path.join(base_dir, f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    async def batch_extract_conversations(self, ticket_ids: list):
        conversations = {}
        cached = {tid: self._conversation_cache.get(tid) for tid in ticket_ids}
        missing_tickets = [tid for tid, convo in cached.items() if convo is None]
        conversations.update({tid: convo for tid, convo in cached.items() if convo is not None})

        for ticket_id in missing_tickets:
            try:
                extractor = ConversationExtractor(self.bq, ticket_id)
                raw = extractor.get_convo_str()
                parsed = extractor.parse_conversation(raw)
                stats = extractor.convo_stats(parsed)

                conversation_data = {
                    "raw": raw,
                    "parsed": parsed,
                    "stats": stats
                }

                conversations[ticket_id] = conversation_data
                self._conversation_cache[ticket_id] = conversation_data
            except Exception as e:
                logging.error(f"Failed to extract conversation for ticket {ticket_id}: {e}")
                conversations[ticket_id] = None
                self._conversation_cache[ticket_id] = None

        return conversations

    def fetch_labeled(self, ticket_id: str = None, limit: int = 50) -> pd.DataFrame:
        """Queries human labeled (training) dataset.
        
        Args:
            `ticket_id` (int): Specific ticket ID to query.
            `limit` (int): The number of ticket IDs (Default: `50`).

        Returns:
            `pd.DataFrame`
        """
        query = """
        SELECT ticket_id, top_intent_h
        FROM `{}.{}.labeled_intent`
        """.format(self.bq.client.project, self.bq.dataset_id)
        if ticket_id is not None:
            query += f"\nWHERE ticket_id = '{ticket_id}'"
        elif limit is not None:
            query += f"\nLIMIT {limit}"
        return self.bq.execute_query(query)

    async def process_ticket(self, ticket_ids: list, rubric_intent: str, max_concurrent: int = 10):
        semaphore = asyncio.Semaphore(max_concurrent)
        async def run_pipeline(ticket_id: str):
            logging.info(f"Ticket ID: {ticket_id}")
            async with semaphore:
                try:
                    extractor = ConversationExtractor(self.bq, ticket_id)
                    raw = await asyncio.to_thread(extractor.get_convo_str)
                    parsed = await asyncio.to_thread(extractor.parse_conversation, raw)
                    stats = await asyncio.to_thread(extractor.convo_stats, parsed)

                    pipeline = ConversationPipeline(
                        self.openai,
                        parsed,
                        stats,
                        self.rubric_prompt,
                        rubric_intent,
                        model=self.model
                    )
                    result = await pipeline.run()
                    return ticket_id, result, pipeline
                except Exception as e:
                    logging.error(f"Erorr on ticket {ticket_id}: {e}")
                    return ticket_id, None, None

        tasks = [run_pipeline(ticket_id) for ticket_id in ticket_ids]
        results = await asyncio.gather(*tasks)

        return {
            ticket_id: {
                "result": result,
                "pipeline": pipeline
            }
            for ticket_id, result, pipeline in results
        }

    async def evaluate(self, human_label: pd.DataFrame, ticket_ids: list, current_rubric: str, max_concurrent: int):
        try:
            initial_results = await self.process_ticket(ticket_ids, current_rubric, max_concurrent)
            initial_transformed = FeedbackLoop.transform_scorecard_results(initial_results)
            initial_df = pd.DataFrame(initial_transformed).rename(columns={"top_intent": "top_intent_llm"})
            initial_df = pd.merge(initial_df, human_label, on="ticket_id")
            initial_df["match"] = initial_df["top_intent_llm"] == initial_df["top_intent_h"]
            initial_accuracy = initial_df["match"].mean()

            return {
                "current_results": initial_results,
                "current_df": initial_df,
                "current_accuracy": initial_accuracy
            }
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return None

    # Rubric methods
    async def refine_rubric(self, mismatches: list, current_rubric: str):
        if not mismatches:
            return current_rubric

        ticket_ids = [m["ticket_id"] for m in mismatches]
        conversations = await self.batch_extract_conversations(ticket_ids)
        
        try:
            iteration = len(self.rubric_history)

            processor = RubricProcessor(
                self.openai,
                SUMMARIZE_RUBRIC_PROMPT,
                IDENTIFY_RUBRIC_ISSUES_PROMPT,
                MODIFY_RUBRIC_PROMPT,
                self.temperature,
                self.model
            )
            
            logging.info("========== Identifying Rubric Issues ==========")
            identified_issues = await processor.identify_rubric_issues_for_ticket(mismatches, conversations, current_rubric)

            if not identified_issues or all(item["issue"] is None for item in identified_issues):
                logging.warning("No issues identified. Using original rubric...")
                return current_rubric

            logging.info("========== Summarizing Identified Rubric Issues ==========")
            summarized_issues = await processor.summarize_generated_rubric_issues(identified_issues)

            logging.info("========== Saving Rubric Artifacts ==========")
            file_data = [
                {"file_name": "issues_v{iteration}.json", "data": identified_issues},
                {"file_name": "issues_summary_v{iteration}.txt", "data": summarized_issues}
            ]
            run_dir = FeedbackLoop.get_run_dir()
            processor.save_rubric_artifacts(run_dir, iteration, file_data)

            logging.info("========== Modifying Rubric ==========")
            modified_rubric = await processor.modify_rubric(current_rubric, identified_issues)

            logging.info("========== Validating Modified Rubric ==========")
            validated_rubric = processor.clean_response(modified_rubric)

            if not validated_rubric or not validated_rubric.strip():
                logging.warning("Rubric modification failed. Using original rubric...")
                return current_rubric
            return validated_rubric
        except Exception as e:
            logging.error(f"Error in rubric refinement: {e}")
            logging.error("Returning original rubric...")
            return current_rubric

    def save_rubric(self, rubric: str, iteration: int, timestamp: str = None):
        os.makedirs("rubrics/evolution", exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        filename = f"rubrics/evolution/rubric_v{iteration}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(rubric)

        rubric_info = {
            "iteration": iteration,
            "timestamp": timestamp,
            "content": rubric,
            "filename": filename
        }

        self.rubric_history.append(rubric_info)
        return filename

    async def run_feedback_loop(self):
        logging.info("========== STARTUP ==========")
        logging.info(f"Number of tickets: {self.limit}")
        logging.info(f"Number of iterations: {self.iterations}")
        human_label = self.fetch_labeled(limit=self.limit)
        if human_label.empty:
            logging.warning("No labeled data found.")
            return

        iteration = 0
        ticket_ids = human_label["ticket_id"].to_list()

        logging.info("Retrieving initial rubric...")
        current_rubric = self.rubric_intent
        logging.info("Saving initial rubric...")
        self.save_rubric(current_rubric, iteration)
        logging.info("Initial rubric saved!")

        logging.info("Evaluating using base rubric...")
        eval = await self.evaluate(human_label, ticket_ids, current_rubric, MAX_CONCURRENT)

        initial_results = None
        initial_accuracy = None
        initial_df = None

        best_results = None
        best_accuracy = None
        best_rubric = None

        logging.info("========== Initial Evaluation Using Base Rubric ==========")
        if eval:
            logging.info("Done evaluating!")
            initial_results = eval["current_results"]
            initial_accuracy = eval["current_accuracy"]
            initial_df = eval["current_df"]

        if initial_accuracy >= 0.9:
            logging.info(f"Thresheld met: {initial_accuracy:.2%}. Skipping rubric refinement and accepting current rubric.")
            return initial_results

        current_results = initial_results
        current_accuracy = initial_accuracy
        current_df = initial_df

        logging.info("========== Current Stats ==========")
        logging.info(f"Current Accuracy: {current_accuracy:.2%}")

        best_results = current_results
        best_accuracy = current_accuracy
        best_rubric = current_rubric

        new_results = None
        new_accuracy = None
        new_df = None

        logging.info("========== Best Stats ==========")
        logging.info(f"Best Accuracy: {best_accuracy:.2%}")

        improvement_threshold = 0.05
        consecutive_no_improvement = CONSECUTIVE_NO_IMPROVEMENT
        max_consecutive_no_improvement = MAX_CONSECUTIVE_NO_IMPROVEMENT

        logging.info(f"Did not meet improvement threshold ({improvement_threshold:+.2%}). Performing Feedback loop...")
        while iteration < self.iterations:
            iteration += 1
            logging.info(f"========== Iteration#{iteration} ==========")
            logging.info("Collecting mismatches from current results...")
            mismatches = [
                {"ticket_id": row["ticket_id"], "llm": row["top_intent_llm"], "human": row["top_intent_h"]}
                for _, row in current_df.iterrows()
                if not row["match"]
            ]
            logging.info(f"Current accuracy: {current_accuracy:.2%} | Mismatches: {len(mismatches)}")

            if not mismatches:
                logging.info("🎉 Perfect match achieved! No more mismatches to resolve.")
                break

            logging.info("Refining rubric based on mismatches...")
            try:
                new_rubric = await self.refine_rubric(mismatches, best_rubric)
                if not new_rubric or new_rubric.strip() == best_rubric.strip():
                    logging.warning("Rubric refinement produced no changes. Stopping iteration...")
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= max_consecutive_no_improvement:
                        logging.info("No rubric changes for multiple iterations. Stopping iteration...")
                        break
                    continue
            except Exception as e:
                logging.error(f"Failed to refine rubric: {e}")
                break

            self.save_rubric(new_rubric, iteration)
            logging.info("New rubric saved!")

            logging.info("========== Testing Refined Rubric ==========")
            test = await self.evaluate(human_label, ticket_ids, new_rubric, MAX_CONCURRENT)
            if test:
                new_results = test["current_results"]
                new_accuracy = test["current_accuracy"]
                new_df = test["current_df"]

            improvement = new_accuracy - current_accuracy
            logging.info(f"Collecting accuracy #{iteration}...")
            self.accuracies.append(new_accuracy)
            logging.info(f"New accuracy: {new_accuracy:.2%} | Improvement: {improvement:+.2%}")

            logging.info("========== Evaluating Improvement ==========")
            if new_accuracy > best_accuracy + improvement_threshold:
                logging.info("Significant improvement detected using the refined rubric.")
                logging.info("Using new and refined rubric...")
                current_rubric = new_rubric
                logging.info("Done!")
                current_df = new_df
                current_results = new_results
                consecutive_no_improvement = 0

                best_accuracy = new_accuracy
                best_rubric = new_rubric
                best_results = new_results
            elif new_accuracy >= best_accuracy:
                logging.info("Minor improvement detected. Rolling back to best rubric...")
                consecutive_no_improvement = 0
                current_rubric = best_rubric
                current_results = new_results
                current_df = new_df

                best_accuracy = new_accuracy
                best_results = new_results
            else:
                logging.warning("Performance did not improve. Rejecting rubric...")
                consecutive_no_improvement += 1
                if best_rubric != current_rubric:
                    current_rubric = best_rubric
                    current_results = best_results
                    current_df = pd.DataFrame(FeedbackLoop.transform_scorecard_results(best_results)).rename(columns={"top_intent": "top_intent_llm"})
                    current_df = pd.merge(current_df, human_label, on="ticket_id")
                    current_df["match"] = current_df["top_intent_llm"] == current_df["top_intent_h"]
                    logging.info(f"Rolled back to best rubric (accuracy: {best_accuracy:.2%})")

                logging.info("========== Checking Matches ==========")
                if new_df["match"].all():
                    current_results = new_results
                    current_df = new_df
                    break

        final_accuracy = current_df["match"].mean() if current_df is not None else best_accuracy
        total_improvement = final_accuracy - initial_accuracy
        logging.info(f"\n========== Feedback Loop Complete ==========")
        logging.info(f"Iterations completed: {iteration}")
        logging.info(f"Initial accuracy: {initial_accuracy:.2%}")
        logging.info(f"Final accuracy: {final_accuracy:.2%}")
        logging.info(f"Total improvement: {total_improvement:+.2%}")
        logging.info(f"Accuracy per iteration: {[float(a) for a in self.accuracies]}")

        if final_accuracy < best_accuracy:
            logging.info(f"Using best performing rubric (accuracy: {best_accuracy:.2%})")
            current_results = best_results
            current_rubric = best_rubric

            final_transformed = FeedbackLoop.transform_scorecard_results(best_results)
            final_df = pd.DataFrame(final_transformed).rename(columns={"top_intent": "top_intent_llm"})
            final_df = pd.merge(final_df, human_label, on="ticket_id")
            final_df["match"] = final_df["top_intent_llm"] == final_df["top_intent_h"]
            final_accuracy = final_df["match"].mean()

        final_transformed = FeedbackLoop.transform_scorecard_results(current_results)
        logging.info(f"Raw: {final_transformed}")
        for result in final_transformed:
            logging.info(f"Ticket {result['ticket_id']}: {result['top_intent']}")

        return {
            "current_results": current_results,
            "iteration": iteration,
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "total_improvement": total_improvement,
            "accuracies": self.accuracies
        }