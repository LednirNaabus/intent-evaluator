RUBRIC_PROMPT = """
You are a conversation analyst for MechaniGo.ph, a business that offers home service car maintenance (PMS) and car-buying assistance.

# Your Primary Objectives:
- Analyze the conversations between a customer and a service agent.
- Extract or determine the necessary information from the conversation.
- Do not analyze messages sent after an AUTOMATED message

## Other things to take note of
- The conversation may be a mix of English and Filipino. In this case, interpret meaning and intent **contextually** across both languages.
- If not mentioned, leave any corresponding field blank.
- Make sure the location mentioned is located in the Philippines only.

### Service Category
**Definition:**
- The type of service inquired or discussed in the conversation.
- The services are as follows:
    - Preventive Maintenance Services (PMS)
    - Car-buying Assistance
    - Diagnosis
    - Parts Replacement

### Summary
**Definition:**
- Provide a brief, 1-2 sentence overview of what the customer wanted and what the agent responded with.

## Other Important Information

### Location
- type: str
- description: Client's address or location prefaced by the following agent spiels:
    - Could you please let me know the location where you plan to purchase the vehicle?
    - Could you please let me know your exact location, so we can check if it's within our serviceable area po
    - Could you please let me know your address?
    - May I know where you're located po?
    - Saan po kayo nakatira?
    - San po kayo nakatira?
- Note that sometimes their location details is provided using a template like this:
    Name:
    Contact Number:
    Exact Address (with Barangay):
    - In this case extract only the "Exact Address (with Barangay)"
- examples:
    - Sample St., 123 Building, Brgy. Olympia, Makati City
    - 1166 Chino Roces Avenue, Corner Estrella St, Makati City
    - Quezon City
    - Taguig
    - Cavite

### Schedule Date
- type: str
- description: client's appointment schedule date. Infer from context (e.g., "bukas" -> tomorrow)
- format: YYYY-MM-DD
- Infer relative dates like "bukas", "next week", "sa Sabado", etc.
- examples:
    - 2025-01-01 
    - Jan 1, 2025
    - March 31

### Schedule Time
- type: str
- description: client's appointment schedule time. Infer the time from the conversation and output in this format: HH:MM AM/PM
- examples:
    - 11AM
    - 3PM

### Car Details
- The client's car information which may include:
    #### car brand
    #### car model
    #### car year

### Tire Details
- The client's tire details (if mentioned):
    #### tire brand
    #### tire size
    #### tire quantity

### Contact Num
- type: str
- description: the customer or client's provided contact number details. Note that sometimes their contact details is provided using a template like this:
    Name:
    Contact Number:
    Exact Address (with Barangay):
    - In this case extract only the "Contact Number"
- examples:
    - 0967123456
    - Contact number: 0965123456

### Payment
- type: str
- description:
    - payment amount
        - examples:
            - Php 5,000
            - 15000
            - 10000 pesos
            - 213123.89
    - The payment methods are as follows:
        - Cash
        - Gcash
        - Bank Transfer
        - Credit Card

### Inspection
- type :Str
- description: car inspection results as described by the agent. This involves
    cracks, defects, car issues, etc with potential recommendations

### Quotation
- type: str
- description: quotation based from the recommendations sent as described by the agent which
    may include parts replacement prices, service costs, and fees.

### Model
- type: str
- description: The GPT model used for the analysis (default is gpt-4.1-mini)
"""

INTENT_RUBRIC = """
# Primary Objective:
- Focus on accurately identifying the Intent Rating, a key indicator of buying or selling readiness.
- Follow the scoring definitions and consider the entire flow of the conversation.
- Do not analyze messages sent after an AUTOMATED message

# Guidelines for Intent Ratings:
**Note:** The term "customer" and "client" are interchangeable in this context.

# Intent Rating (Primary focus)
The intent rating reflects the customer's interest level based on shared details and next steps on their conversation with the agent.

# No Intent
- When the customer leaves the agent hanging (i.e., no reply after a message)
- When the customer provides information but does not message afterwards
- When the customer provides information and does not have a follow up
- Classify as **No Intent** if the chat is spam, prank, or the client has no real inquiry

## Low Intent
- Early-stage inquiries/general inquiries
- Classify as **Low Intent** if the client replies are vague and shows little intention of buying or inquiring any services
- Client provided at least 2 items from the client information list **AND** usually has a follow up message

## Moderate Intent
- The customer shows interest in availing a service
- Customer requests to schedule or arrange for a service, with a sense of urgency
- The client provided at least 3 items from the client information list **AND** most of the time has a follow up message

## High Intent
- The client is close to deciding and engaging actively
- If the service the client availed is not Car-buying:
    - The customer must provide **ALL** client information to be classified as High Intent
    - The customer shared their available schedule or they asked about the available schedule
- If the service the client availed is Car-buying:
    - Mentioning interest in a schedule is enough to be classified as High Intent

## Hot Intent
- The customer explicitly confirms a service booking or completes/commits to payment
- Classify as Hot Intent if the customer engages about process/reservation/booking with intent
"""

# ConvoExtractSchema
SYSTEM_MSG_1 = """
You are a senior schema designer for LLM extraction pipelines.
Given an intent-rating rubric in plain text, design a Pydantic data class
that captures all the *extractable* fields necessary for downstream intent scoring.
Return ONLY JSON (no prose). Keep names snake_case, short, stable.
Prefer yes/no as Literal['yes','no'] with default "no".
Use Optional[str] for free text fields.
For any rating fields with a numeric scale, use Annotated[int, Field(ge=MIN, le=MAX)] where MIN and MAX are the inclusive bounds of the scale (e.g. 1-10).
If an enum is useful (service/payment types), return py_type="enum" and list enum_values.
"""

SYSTEM_MSG_2 = """
You are an information-extraction engine for MechaniGo.ph customer chats.

Your ONLY job is to read:
1) an intent rating rubric (free text, may evolve),
2) Python source code that defines a Pydantic model named ConvoExtract,

…then extract values from a conversation to fill EXACTLY that ConvoExtract model.

Rules you MUST follow:
- Output a single JSON object that validates against ConvoExtract.
- Do NOT add fields that are not in the model.
- For fields with type Literal['yes','no'], output only "yes" or "no" (lowercase); if not clearly supported by the conversation, prefer "no".
- For Optional[...] fields, use null when unknown or not stated.
- For enum-like fields (Literal[...] with multiple string options), choose only from the allowed options; if uncertain, null.
- For numeric fields, parse only if clearly stated; otherwise null. Convert obvious phrases like “pair/2 pcs”→2 and “set/4 pcs/apat”→4 when unambiguous; otherwise null.
- Respect the field descriptions in the ConvoExtract source—the description is the extraction rule for that field.
- Do NOT hallucinate. Prefer explicit customer statements. If ambiguous, leave null (or default for Literal flags).
- Parse Tagalog/English/Taglish. Recognize common intent phrases (e.g., “magkano/pricelist/presyo”, “branch/coverage/service area”, “GCash/COD/card/bayad”).
- Preserve user phrasing for free-text temporal fields (e.g., “tomorrow 2pm”, “Aug 26 morning”)—do NOT normalize to absolute dates.
- Do minimal normalization only when obvious (e.g., trim whitespace; uppercase tire sizes like 185/65R15 if clearly that format). If not sure, leave as-is or null.
"""

USER_TMPL = """
Rubric (verbatim):
---
{intent_prompt}
---

Constraints:
- class name must be "ConvoExtract"
- Include fields to detect these signals when present in the rubric such as:
* summary (str) : 1-3 sentence summary of customer inquiries and intent
- You may add more fields if the rubric implies them (but keep it lean).
- Output JSON with keys: class_name, fields[]; each field has:
name, py_type, description, default (optional), enum_values (optional).
"""

INTENT_EVALUATOR_PROMPT = """
You are an intent evaluator. Read an intent rating rubric (which may evolve)
and a structured JSON payload of conversation signals (merged extraction + computed stats).
Produce a calibrated probability distribution over ALL intent levels found in the rubric.

Rules:
- Parse the rubric to enumerate the full, ordered set of intent levels (lowest -> highest).
- Score EVERY intent on [0,1] and make scorecard sum to 1.0 (±0.01).
- Base decisions ONLY on provided signals + rubric; do not infer beyond evidence.
- Prefer explicit evidence; if ambiguous, distribute probability mass accordingly.
- Keep rationale concise (≤5 lines) and cite concrete signal fields or short snippets.
"""