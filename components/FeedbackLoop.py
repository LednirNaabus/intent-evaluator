"""
The feedback loop can be considered as the training phase for improving the rubric criteria
for classifying the intent rating of conversations between a client and an agent.
"""
from components import ConversationExtractor, ConversationPipeline
from clients import BigQueryClient, OpenAIClient
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_ITER = 1

class FeedbackLoop:
    def __init__(
        self,
        openai: OpenAIClient,
        bq: BigQueryClient,
        ticket_id: str,
        rubric_prompt: str,
        rubric_intent: str,
        model: str = "gpt-4.1-mini"
    ):
        self.openai = openai
        self.bq = bq
        self.ticket_id = ticket_id
        self.rubric_prompt = rubric_prompt
        self.rubric_intent = rubric_intent
        self.model = model

    # For batches
    @staticmethod
    def transform_scorecard_results(results: dict, ticket_id: str = None):
        records = []

        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            for ticket, details in results.items():
                records.append({
                    "ticket_id": ticket,
                    "top_intent": details["top_intent"]
                })
        elif ticket_id is not None and isinstance(results, dict):
            records.append({
                "ticket_id": ticket_id,
                "top_intent": results["top_intent"]
            })
        else:
            raise ValueError("Unsupported format. Check the results again.")
        return records

    # For single ticket/scorecard
    @staticmethod
    def transform_scorecard_result(ticket_id: str, details: dict):
        return [{
            "ticket_id": ticket_id,
            "top_intent": details["top_intent"]
        }]

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

    # get labeled
    def get_labeled(self, ticket_id: str, limit: int = 50) -> pd.DataFrame:
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

    async def process_ticket(self, rubric_intent: str):
        logging.info(f"Ticket ID: {self.ticket_id}")
        extractor = ConversationExtractor(self.bq, self.ticket_id)
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
            return result, pipeline
        except Exception as e:
            logging.error(f"Exception occurred while analyzing ticket: {e}")
            return self.ticket_id, None

    # run the pipeline using base prompt
    # compare the intent ratings
    # iterate and refine the prompts if there are mismatches
    async def run_feedback_loop(self):
        human_label = self.get_labeled(ticket_id=self.ticket_id)
        if human_label.empty:
            logging.warning(f"No labeled data found for {self.ticket_id}")
            return

        rubric = self.rubric_intent
        results = None
        # Main feedback loop
        for i in range(MAX_ITER):
            results, pipeline = await self.process_ticket(rubric)
            transformed = FeedbackLoop.transform_scorecard_result(self.ticket_id, results)
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
            with open(f"rubrics/rubric_v{i+1}.txt", "w") as f:
                f.write(rubric)
        else:
            print("\nMax iterations reached.")

        final_transformed = FeedbackLoop.transform_scorecard_result(self.ticket_id, results)
        print("========== Final Results ==========\n")
        print(final_transformed)
        final_df = pd.DataFrame(final_transformed).rename(columns={"top_intent": "top_intent_llm"})
        final_df = pd.merge(final_df, human_label, on="ticket_id")
        final_df["match"] = final_df["top_intent_llm"] == final_df["top_intent_h"]

        print("\n========== DataFrame ==========\n")
        print(final_df)

        return results, pipeline