from components import ConversationExtractor, ConversationPipeline
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

MAX_ITER = 3

class FBL:
    def __init__(
        self,
        openai: OpenAIClient,
        bq: BigQueryClient,
        limit: int,
        rubric_prompt: str,
        rubric_intent: str,
        iterations: int = MAX_ITER,
        model: str = "gpt-4.1-mini"
    ):
        self.openai = openai
        self.bq = bq
        self.limit = limit
        self.rubric_prompt = rubric_prompt
        self.rubric_intent = rubric_intent
        self.iterations = iterations
        self.model = model
        self.rubric_history = []
        self._conversation_cache = {}

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

    def get_labeled(self, ticket_id: str = None, limit: int = 50) -> pd.DataFrame:
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

    # process tickets
    async def process_ticket(self, ticket_ids: list, rubric_intent: str):
        async def run_pipeline(ticket_id: str):
            logging.info(f"Ticket ID: {ticket_id}")
            extractor = ConversationExtractor(self.bq, ticket_id)
            raw = extractor.get_convo_str()
            parsed = extractor.parse_conversation(raw)
            stats = extractor.convo_stats(parsed)

            pipeline = ConversationPipeline(
                self.openai,
                parsed,
                stats,
                self.rubric_prompt,
                rubric_intent,
                model=self.model
            )
            try:
                result = await pipeline.run()
                return ticket_id, result, pipeline
            except Exception as e:
                logging.error(f"Exception occurred while analyzing ticket: {e}")
                return ticket_id, None, pipeline

        tasks = [run_pipeline(ticket_id) for ticket_id in ticket_ids]
        results = await asyncio.gather(*tasks)
        return {
            ticket_id: {
                "result": result,
                "pipeline": pipeline
            }
            for ticket_id, result, pipeline in results
        }

    # Rubric methods
    # Per ticket ID basis
    async def identify_rubric_issues(self, mismatches: list, current_rubric: str):
        logging.info("Identifying rubric issues...")
        if not mismatches:
            return ""

        ticket_ids = [m["ticket_id"] for m in mismatches]
        conversations = await self.batch_extract_conversations(ticket_ids)

        conversation_blocks = []
        for m in mismatches:
            ticket_id = m["ticket_id"]
            convo_data = conversations.get(ticket_id)

            if convo_data and convo_data.get("raw"):
                convo_block = (
                    f"Conversation for {ticket_id}:\n"
                    f"{convo_data['raw']}\n"
                    f"- LLM predicted: {m['llm']}\n"
                    f"- Human label: {m['human']}\n"
                    f"- Conversation stats: {convo_data.get('stats', 'N/A')}\n"
                )
            else:
                convo_block = (
                    f"Conversation for {ticket_id}:\n"
                    f"[Conversation extraction failed]\n"
                    f"- LLM predicted: {m['llm']}\n"
                    f"- Human label: {m['human']}\n"
                )

            conversation_blocks.append(convo_block)
        examples = "\n\n".join(conversation_blocks)
        prompt = """
        Below is a rubric for classifying intent ratings from a conversation between a client and a sales agent. Following the rubric are several examples where the LLM's classification did not match the human-labeled ground truth.

        Current Rubric:
        ---
        {}
        ---

        Mismatched Conversations:
        ---
        {}
        ---

        Task:
        - Based on the mismatches, identify specific parts of the rubric that are unclear, incomplete, or misleading.
        - Return the identified issues as a JSON object. 
        """.format(current_rubric, examples)
        try:
            completion = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intent rating rubric evaluator. Analyze the current rubric and identify potential issues that may have caused these mismatches."
        
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error during rubric issue identification: {e}")
            return ""

    async def modify_rubric(self, current_rubric: str, identified_issues: dict):
        prompt = """
        Below is a rubric (which may evolve) for classifying intent ratings, along with a summary of issues identified from previous classification mismatches.

        Current/Previous Rubric (evolving):
        ---
        {}
        ---

        Identified Issues (in JSON format):
        ---
        {}
        ---
        Task:
        - Modify the rubric to resolve the identified issues from the previous version of the rubric.
        - ONLY return the modified rubric. Do not include the previous rubric and the identified issues.
        """.format(current_rubric, identified_issues)
        try:
            completion = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at writing precise intent rating rubrics. Read the current current rubric and the given identified issues."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error during rubric modification: {e}")
            return current_rubric

    async def refine_rubric(self, mismatches: list, current_rubric: str):
        if not mismatches:
            return current_rubric
        try:
            logging.info("========== Identifying Rubric Issues ==========")
            identified_issues = await self.identify_rubric_issues(mismatches, current_rubric)

            if not identified_issues or not identified_issues.strip():
                logging.warning("No issues identified. Using original rubric...")
                return current_rubric

            logging.info("Issues found in rubric!")
            logging.info("========== Modifying Rubric ==========")
            modified_rubric = await self.modify_rubric(current_rubric, identified_issues)

            if not modified_rubric or not modified_rubric.strip():
                logging.warning("Rubric modification failed. Using original rubric...")
                return current_rubric
            return modified_rubric
        except Exception as e:
            logging.error(f"Error in rubric refinement: {e}")
            logging.error("Returning original rubric...")
            return current_rubric

    def save_rubric(self, rubric: str, iteration: int, timestamp: str = None):
        os.makedirs("rubrics", exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        filename = f"rubrics/rubric_v{iteration}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(rubric)

        rubric_info = {
            "iteration": iteration,
            "timestamp": timestamp,
            "content": rubric,
            "filename": filename,
        }

        self.rubric_history.append(rubric_info)
        return filename

    async def evaluate(self, human_label: pd.DataFrame, ticket_ids: list, current_rubric: str):
        logging.info("Running initial evaluation using base rubric...")
        try:
            initial_results = await self.process_ticket(ticket_ids, current_rubric)
            initial_transformed = FBL.transform_scorecard_results(initial_results)
            initial_df = pd.DataFrame(initial_transformed).rename(columns={"top_intent": "top_intent_llm"})
            initial_df = pd.merge(initial_df, human_label, on="ticket_id")
            initial_df["match"] = initial_df["top_intent_llm"] == initial_df["top_intent_h"]
            initial_accuracy = initial_df["match"].mean()
            
            return {
                "current_results": initial_results,
                "current_df": initial_df,
                "accuracy": initial_accuracy
            }
        except Exception as e:
            logging.error(f"Failed initial evaluation: {e}")
            return None

    # the loop
    async def run_feedback_loop(self):
        # Initialization
        # - load base rubric (from config or user input in Streamlit form)
        # - fetch human labels from BQ
        # - set max iterations
        human_label = self.get_labeled(limit=self.limit)
        if human_label.empty:
            logging.warning("No labeled data found.")
            return
        
        ticket_ids = human_label["ticket_id"].to_list()
        current_rubric = self.rubric_intent # set base rubric

        eval = await self.evaluate(human_label, ticket_ids, current_rubric)
        if eval:
            initial_results = eval["current_results"]
            initial_accuracy = eval["accuracy"]
            initial_df = eval["current_df"]

        # if initial_accuracy >= 0.85 threshold met, return initial_results
        if initial_accuracy >= 0.85:
            logging.info(f"Threshold met: {initial_accuracy:.2%}. Skipping rubric refinement and accepting current rubric.")
            return initial_results

        logging.info("========== Initial Accuracy ==========")
        print(f"{initial_accuracy:.2%}")
        # Otherwise:
        # initialize variables
        iteration = 0
        improvement_threshold = 0.05

        current_results = initial_results
        current_df = initial_df

        best_results = initial_results
        best_accuracy = initial_accuracy
        best_rubric = current_rubric

        consecutive_no_improvement = 0
        max_consecutive_no_improvement = self.iterations + 1
        
        # while iteration < self.iterations:
        #     collect mismatches
        #     if not mismatches:
        #         break
        #     identify rubric issues
        #     refined_rubric = modify rubric

        #     new_accuracy, new_df, new_results = evaluate(modified_rubric) # test refined rubric
        logging.info("Did not meet improvement threshold. Performing feedback loop.")
        while iteration < self.iterations:
            iteration += 1
            logging.info(f"========== Iteration #{iteration} ==========")
            logging.info("Collecting mismatches from current results...")
            mismatches = [
                {"ticket_id": row["ticket_id"], "llm": row["top_intent_llm"], "human": row["top_intent_h"]}
                for _, row in initial_df.iterrows()
                if not row["match"]
            ]
            current_accuracy = initial_df["match"].mean()
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
            test = await self.evaluate(human_label, ticket_ids, new_rubric)
            if test:
                new_results = test["current_results"]
                new_accuracy = test["accuracy"]
                new_df = test["current_df"]
            
            improvement = new_accuracy - current_accuracy
            logging.info(f"New accuracy: {new_accuracy:.2%} | Improvement: {improvement:+.2%}")

            logging.info("Evaluating improvement...")
            if improvement > improvement_threshold:
                logging.info("Significant improvement detected using the refined rubric.")
                logging.info("Using new and refined rubric...")
                current_rubric = new_rubric
                logging.info("Done!")
                current_df = new_df
                current_results = new_results
                consecutive_no_improvement = 0

                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_rubric = new_rubric
                    best_results = new_results
                    logging.info(f"New best accuracy: {best_accuracy:.2%}")
            elif improvement >= 0:
                logging.info("Minor improvement detected.")
                current_rubric = new_rubric
                current_results = new_results
                current_df = new_df
                consecutive_no_improvement += 1

                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_rubric = new_rubric
                    best_results = new_results
            else:
                logging.warning("Performance did not improve. Rejecting rubric...")
                consecutive_no_improvement += 1
                if best_rubric != current_rubric:
                    current_rubric = best_rubric
                    current_results = best_results
                    current_df = pd.DataFrame(FBL.transform_scorecard_results(best_results)).rename(columns={"top_intent": "top_intent_llm"})
                    current_df = pd.merge(current_df, human_label, on="ticket_id")
                    current_df["match"] = current_df["top_intent_llm"] == current_df["top_intent_h"]
                    logging.info(f"Rolled back to best rubric (accuracy: {best_accuracy:.2%})")

            if consecutive_no_improvement >= max_consecutive_no_improvement:
                logging.info("No meaningful improvement over multiple iterations. Stopping...")
                break

            logging.info("Checking matches...")
            if new_df["match"].all():
                logging.info("Perfect match achieved")
                current_results = new_results
                current_df = new_df
                break

        final_accuracy = current_df["match"].mean() if "current_df" in locals() else best_accuracy
        total_improvement = final_accuracy - initial_accuracy
        logging.info(f"\n========== Feedback Loop Complete ==========")
        logging.info(f"Iterations completed: {iteration}")
        logging.info(f"Initial accuracy: {initial_accuracy:.2%}")
        logging.info(f"Final accuracy: {final_accuracy:.2%}")
        logging.info(f"Total improvement: {total_improvement:+.2%}")

        if iteration == self.iterations:
            logging.info("Maximum iterations reached")

        if final_accuracy < best_accuracy:
            logging.info(f"Using best performing rubric (accuracy: {best_accuracy:.2%})")
            current_results = best_results
            current_rubric = best_rubric

            final_transformed = FBL.transform_scorecard_results(best_results)
            final_df = pd.DataFrame(final_transformed).rename(columns={"top_intent": "top_intent_llm"})
            final_df = pd.merge(final_df, human_label, on="ticket_id")
            final_df["match"] = final_df["top_intent_llm"] == final_df["top_intent_h"]
            final_accuracy = final_df["match"].mean()

        final_transformed = FBL.transform_scorecard_results(current_results)
        logging.info("========== Final Results ==========")
        for result in final_transformed:
            logging.info(f"Ticket {result['ticket_id']}: {result['top_intent']}")

        return current_results