from config import DATASET_NAME, BQ_CLIENT, MNL_TZ
from typing import Optional, Literal
from google.cloud import bigquery

import pandas as pd

class BigQueryClient:
    def __init__(self, client: bigquery.Client = BQ_CLIENT):
        self.client = client
        self.dataset_id = DATASET_NAME

    def execute_query(self, query: str, return_data: bool = True) -> Optional[pd.DataFrame]:
        query_job = self.client.query(query)
        if return_data:
            df = query_job.to_dataframe()
            return df
        else:
            query_job.result()
            return None

    def recent_tickets(
        self,
        table_name: Literal["tickets", "messages"],
        date_filter: str = "datecreated",
        limit: int = 10
    ) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC").astimezone(MNL_TZ)
        date = now - pd.Timedelta(hours=6)
        start = date.floor('h')
        end = start + pd.Timedelta(hours=6) - pd.Timedelta(seconds=1)
        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")

        if table_name == "tickets":
            select_clause = "id, owner_name, agentid"
            where_conditions = []
        elif table_name == "messages":
            select_clause = "DISTINCT ticket_id"
            where_conditions = ["message_format = 'T'"]
        else:
            raise ValueError(f"The table name '{table_name}' not found.")

        where_clauses = [
            f"{date_filter} >= '{start_str}'",
            f"{date_filter} < '{end_str}'"
        ]

        where_clauses.extend(where_conditions)
        where_clause = " AND ".join(where_clauses)

        query = """
        SELECT {}
        FROM {}.{}.{}
        WHERE {}
        """.format(select_clause, self.client.project, self.dataset_id, table_name, where_clause)

        if limit is not None:
            query += f"\nLIMIT {limit}"
        return self.execute_query(query, return_data=True)