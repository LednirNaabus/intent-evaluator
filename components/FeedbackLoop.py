"""
The feedback loop can be considered as the training phase for improving the rubric criteria
for classifying the intent rating of conversations between a client and an agent.
"""
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

class FeedbackLoop:
    def __init__(
        self,
        openai: OpenAIClient,
        bq: BigQueryClient,
        rubric_prompt: str,
        rubric_intent: str,
        model: str = "gpt-4.1-mini"
    ):
        self.openai = openai
        self.bq = bq
        self.rubric_prompt = rubric_prompt
        self.rubric_intent = rubric_intent
        self.model = model
        self.rubric_history = []

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
    def build_feedback_prompt(mismatches: list, current_rubric: str):
        examples = "\n\n".join(
            f"Conversation {m['ticket_id']}:\n"
            f"- LLM predicted: {m['llm']}:\n"
            f"- Human label: {m['human']}:\n"
            for m in mismatches
        )
        return """
        Update the rubric for classifying the intent rating on a conversation between a client and an agent.

        Here is the current rubric:

        ---
        {}
        ---
        Here are cases where your classification did not align with human labels:
        {}


        Task:
        1. Identify where the rubric may be unclear, incomplete, or misleading.  
        2. Modify the rubric to resolve these mismatches while preserving correct logic for other cases.  
        3. Return ONLY the improved rubric text, not explanations.
        """.format(current_rubric, examples)

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
            "filename": filename
        }
        self.rubric_history.append(rubric_info)
        logging.info(f"Rubric saved: {filename}")
        return filename

    # get labeled
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

    async def refine_rubric(self, mismatches: list, current_rubric: str):
        prompt = FeedbackLoop.build_feedback_prompt(mismatches, current_rubric)
        completion = await self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at writing precise classification rubrics."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

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

    # run the pipeline using base prompt
    # compare the intent ratings
    # iterate and refine the prompts if there are mismatches
    async def run_feedback_loop(self):
        human_label = self.get_labeled(limit=5)
        if human_label.empty:
            logging.warning("No labeled data found.")
            return

        ticket_ids = human_label["ticket_id"].to_list()

        rubric = self.rubric_intent
        results = None
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        self.save_rubric(rubric, 0, timestamp)
        # Main feedback loop
        for i in range(MAX_ITER):
            logging.info(f"Running {MAX_ITER} iterations...")
            results = await self.process_ticket(ticket_ids,rubric)
            print(results)
            transformed = FeedbackLoop.transform_scorecard_results(results)
            llm_scorecard = pd.DataFrame(transformed).rename(columns={"top_intent": "top_intent_llm"})
            df = pd.merge(llm_scorecard, human_label, on="ticket_id")
            df["match"] = df["top_intent_llm"] == df["top_intent_h"]
            print(df)

            mismatches = [
                {"ticket_id": row["ticket_id"], "llm": row["top_intent_llm"], "human": row["top_intent_h"]}
                for _, row in df.iterrows()
                if not row["match"]
            ]

            print(f"\n========== Iteration #{i+1}==========\n")
            print(f"✅ Matches: {df['match'].sum()} | ❌ Mismatches: {len(mismatches)}")

            if not mismatches:
                print("\nAll tickets match. Intent Rubric doesn't need refining.")
                break

            print("Refining rubric...")
            new_rubric = await self.refine_rubric(mismatches, rubric)

            print("========== Refined Rubric ==========\n")
            print(new_rubric)

            print("\nSaving rubric...")
            self.save_rubric(new_rubric, i+1, timestamp)
            rubric = new_rubric
            logging.info("Rubric saved.")
        else:
            print("\nMax iterations reached.")

        final_transformed = FeedbackLoop.transform_scorecard_results(results)
        print("========== Final Results ==========\n")
        print(final_transformed)
        final_df = pd.DataFrame(final_transformed).rename(columns={"top_intent": "top_intent_llm"})
        final_df = pd.merge(final_df, human_label, on="ticket_id")
        final_df["match"] = final_df["top_intent_llm"] == final_df["top_intent_h"]

        print("\n========== DataFrame ==========\n")
        print(final_df)

        return results