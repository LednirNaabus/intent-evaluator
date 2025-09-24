from config import SYSTEM_MODIFY_RUBRIC_PROMPT, UNWANTED_PATTERNS
from components.schemas import RubricIssues
from clients import OpenAIClient
from datetime import datetime
import logging
import json
import os
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RubricProcessor:
    def __init__(
        self,
        openai: OpenAIClient,
        summarize_rubric_prompt: str,
        identify_rubric_prompt: str,
        modify_rubric_prompt: str,
        temperature: float,
        model: str
    ):
        self.openai = openai
        self.summarize_rubric_prompt = summarize_rubric_prompt
        self.identify_rubric_prompt = identify_rubric_prompt
        self.modify_rubric_prompt = modify_rubric_prompt
        self.temperature = temperature
        self.model = model

    @staticmethod
    def parse_identified_rubric_issues(raw_issues: list):
        parsed_results = []
        for item in raw_issues:
            ticket_id = item["ticket_id"]
            issue = RubricIssues.model_validate_json(item["issue"])
            parsed_results.append({
                "ticket_id": ticket_id,
                "issue": issue
            })
        return parsed_results

    @staticmethod
    def clean_response(response: str):
        cleaned = response
        for pattern in UNWANTED_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\n\s*\n", "\n", cleaned).strip()

    async def summarize_generated_rubric_issues(self, issues: list):
        parsed_issues = RubricProcessor.parse_identified_rubric_issues(issues)
        issues_text = "\n".join(
            f"- [{item['ticket_id']}] Section: {item['issue'].section}\n Problem: {item['issue'].problem}"
            for item in parsed_issues
        )
        prompt = self.summarize_rubric_prompt.format(issues_text=issues_text)
        messages = [
            {"role": "system", "content": "You are an expert at summarizing rubric flaws."},
            {"role": "user", "content": prompt}
        ]
        try:
            completion = await self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=800
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Exception occurred while summarizing rubric issues: {e}")
            return None

    async def identify_rubric_issues_for_ticket(self, mismatches: list, conversations: dict, current_rubric: str):
        if not mismatches:
            return []

        results = []
        for m in mismatches:
            ticket_id = m["ticket_id"]
            convo_data = conversations.get(ticket_id)
            convo_block = (
                f"Ticket ID: {ticket_id}\n"
                f"{convo_data['raw'] if convo_data and convo_data.get('raw') else '[Conversation extraction failed]'}\n"
                f"- LLM predicted: {m['llm']}\n"
                f"- Human label: {m['human']}\n"
            )
            prompt = self.identify_rubric_prompt.format(current_rubric=current_rubric, convo_block=convo_block)
            messages = [
                {"role": "system", "content": "You are an intent rubric rating evaluator. Output only rubric issues for this ticket."},
                {"role": "user", "content": prompt}
            ]
            try:
                completion = await self.openai.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=RubricIssues,
                    temperature=self.temperature,
                    max_tokens=600
                )
                parsed = completion.choices[0].message.content
                results.append({"ticket_id": ticket_id, "issue": parsed})
            except Exception as e:
                logging.error(f"OpenAI API error during rubric issue identification: {e}")
                results.append({"ticket_id": ticket_id, "issue": None})
        return results

    async def modify_rubric(self, current_rubric: str, identified_issues: dict):
        prompt = self.modify_rubric_prompt.format(current_rubric=current_rubric, identified_issues=identified_issues)
        messages = [
            {"role": "system", "content": SYSTEM_MODIFY_RUBRIC_PROMPT},
            {"role": "user", "content": prompt}
        ]
        try:
            completion = await self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1500
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error during rubric modification: {e}")
            return current_rubric

    def save_rubric_artifacts(self, run_dir: str, iteration: int, file_data: list, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        for item in file_data:
            file_name = item["file_name"]
            base, ext = os.path.splitext(file_name)
            file_name = f"{base}_{timestamp}{ext}"
            data = item["data"]

            file_path = os.path.join(run_dir, file_name.format(iteration=iteration))
            if file_path.endswith(".json"):
                logging.info(f"Saving issues file: {file_path}")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                logging.info(f"Saved successfully: {file_path}")
            elif file_path.endswith(".txt"):
                logging.info(f"Saving summarized issues file: {file_path}")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(data)
                logging.info(f"Saved successfully: {file_path}")
            else:
                logging.error(f"Unsupported file type: {file_name}. Supported extensions are .json and .txt.")