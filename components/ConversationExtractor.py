from typing import List, Dict, Tuple, Optional, Any
from clients import BigQueryClient
from datetime import datetime
from statistics import mean
import re

class ConversationExtractor:
    def __init__(self, bq_client: BigQueryClient, ticket_id: str = None):
        self.ticket_id = ticket_id
        self.bq_client = bq_client

    def get_convo_str(self) -> str:
        query = """
        SELECT sender_type, message, message_datecreated
        FROM `{}.{}.messages`
        WHERE ticket_id = '{}' 
            AND message_type = 'M' AND message_format = 'T'
        ORDER BY datecreated
        """.format(self.bq_client.client.project, self.bq_client.dataset_id, self.ticket_id)
        df_messages = self.bq_client.execute_query(query)
        string = [
            f"sender: {m['sender_type']}\nmessage: {m['message']}\ndate: {m['message_datecreated']}"
            for _, m in df_messages.iterrows()
        ]
        return "\n\n".join(string)

    @staticmethod
    def parse_conversation(conversation: str) -> List[Dict]:
        pattern = r"sender: (\w+)\nmessage: (.*?)\ndate: (.*?)(?=\nsender:|\Z)"
        matches = re.findall(pattern, conversation, re.DOTALL)
        parsed = [
            {'role': match[0].lower(), 'content': match[1].strip(), 'datetime': match[2]}
            for match in matches
        ]
        return parsed

    @staticmethod
    def count_role(conversation_list: List[Dict[str, Any]], role: str) -> int:
        return sum(1 for r in conversation_list if r.get("role") == role)

    @staticmethod
    def parse_dt(s: str) -> datetime:
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")

    @staticmethod
    def count_exchanges(conversation_list: List[Dict[str, Any]]) -> int:
        exchanges = 0
        i = 0
        while i < len(conversation_list):
            if conversation_list[i].get("role") == "client":
                j, got_reply = i + 1, False
                while j < len(conversation_list) and conversation_list[j].get("role") != "client":
                    if conversation_list[j].get("role") in ("system", "agent"):
                        got_reply = True
                    j += 1
                exchanges += 1 if got_reply else 0
                i = j
            else:
                i += 1
        return exchanges

    @staticmethod
    def get_start_end(conversation_list: List[Dict[str, Any]]) -> Tuple[Optional[datetime], Optional[datetime]]:
        start_str = ConversationExtractor.parse_dt(conversation_list[0]["datetime"]) if conversation_list else None
        end_str = ConversationExtractor.parse_dt(conversation_list[-1]["datetime"]) if conversation_list else None
        return start_str, end_str

    @staticmethod
    def compute_average_response_time(conversation_list: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
        deltas = []
        last_time: Optional[datetime] = None
        for r in conversation_list:
            t = ConversationExtractor.parse_dt(r["datetime"])
            role = r.get("role")
            if role in ("system", "agent"):
                last_time = t
            elif role == "client" and last_time is not None:
                delta = (t - last_time).total_seconds()
                if delta >= 0:
                    deltas.append(delta)
                last_time = None # only count the first reply
        avg_secs = mean(deltas) if deltas else None
        avg_hms = (
            None if avg_secs is None else
            f"{int(avg_secs//3600):02d}:{int((avg_secs%3600)//60):02d}:{int(avg_secs%60):02d}"
        )
        return avg_secs, avg_hms

    @staticmethod
    def convo_stats(conversation_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        recs = sorted(conversation_list, key=lambda r: ConversationExtractor.parse_dt(r["datetime"]))

        # count messages for each sender
        num_agent = ConversationExtractor.count_role(recs, "agent")
        num_system = ConversationExtractor.count_role(recs, "system")
        num_user = ConversationExtractor.count_role(recs, "client")

        # exchanges: user message that gets at least one non-user reply before next user message
        exchanges = ConversationExtractor.count_exchanges(recs)
        start_dt, end_dt = ConversationExtractor.get_start_end(recs)
        avg_secs, avg_hms = ConversationExtractor.compute_average_response_time(recs)

        return {
            "num_agent_messages": num_agent,
            "num_system_messages": num_system,
            "num_user_messages": num_user,
            "num_exchanges": exchanges,
            "start_message_datetime": start_dt.isoformat(sep=" ") if start_dt else None,
            "end_message_datetime": end_dt.isoformat(sep=" ") if end_dt else None,
            "avg_client_response_seconds": avg_secs,
            "avg_client_response_hms": avg_hms
        }