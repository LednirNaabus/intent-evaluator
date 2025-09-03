from config import SYSTEM_MSG_1, USER_TMPL

from pydantic import ValidationError, BaseModel
from components.schemas import SchemaSpec
from clients import OpenAIClient

from typing import Optional, Any
from textwrap import dedent
import types
import json
import sys

class ConvoExtractSchema:
    def __init__(
        self,
        intent_prompt: str,
        model: Optional[str] = "gpt-4o-mini",
        client: OpenAIClient = None
    ):
        self.intent_prompt = intent_prompt
        self.model = model if model is not None else "gpt-4.1-mini"
        self.client = client
        self.system_msg = SYSTEM_MSG_1
        self.user_tmpl = USER_TMPL.format(intent_prompt=self.intent_prompt)

    async def ask_open_ai_for_spec(self):
        prompt = dedent(self.user_tmpl).format(rubric=self.intent_prompt).strip()
        resp = await self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": prompt}
            ],
            text_format=SchemaSpec,
            temperature=0,
            max_output_tokens=5000
        )

        try:
            content_text = resp.output_text
        except Exception:
            # Compatibility: pull first text block if output_text isn't present
            blk = getattr(resp, "output", [])[0]
            content_text = blk.content[0].text  # may vary by SDK version

        try:
            return SchemaSpec.model_validate_json(content_text)
        except ValidationError as ve:
            raise RuntimeError(f"LLM returned invalid spec.\nRaw:\n{content_text}\n\n{ve}") from ve

    def render_pydantic_class(self, spec: type[BaseModel]) -> str:
        lines = []
        lines.append("from __future__ import annotations")
        lines.append("from typing import Optional, Literal")
        lines.append("from pydantic import BaseModel, Field")
        lines.append("")
        lines.append(f"class {spec.class_name}(BaseModel):")
        if not spec.fields:
            lines.append("    pass")
            return "\n".join(lines)

        for f in spec.fields:
            py_type = f.py_type.strip()
            if py_type.lower() == "enum":
                # Convert enum spec to Literal['a','b',...]
                if not f.enum_values:
                    raise ValueError(f"Field {f.name} declared enum but no enum_values provided")
                py_type = "Literal[" + ",".join(repr(v) for v in f.enum_values) + "]"

            # Default handling
            default_expr = "..."  # required by default
            if f.default is not None:
                # For strings, quote; for others let json dumps handle safely
                default_expr = json.dumps(f.default)

            # Field line
            lines.append(
                f"    {f.name}: {py_type} = Field({default_expr}, description={json.dumps(f.description)})"
            )
        return "\n".join(lines)

    async def build_model_class_from_source(self):
        spec = await self.ask_open_ai_for_spec()
        source_code = self.render_pydantic_class(spec)
        mod = types.ModuleType("convo_extract_schema")
        exec(source_code, mod.__dict__)
        sys.modules[mod.__name__] = mod

        if not hasattr(mod, "ConvoExtract"):
            raise ValueError("Generated source did not define ConvoExtract")
        ConvoExtract = getattr(mod, "ConvoExtract")

        # Handle postponed annotations
        try:
            ConvoExtract.model_rebuild()
        except TypeError:
            ConvoExtract.model_rebuild(_types_namespace=mod.__dict__)
        return ConvoExtract