APP_DESC_1 = """
[Intent Evaluator](https://github.com/LednirNaabus/intent-evaluator) is a modular Python pipeline that generates conversation data schemas based on a defined intent criteria.
The Streamlit app provides a user-friendly, interactive interface to explore the pipeline’s output.
"""

APP_DESC_2 = """
This module also serves as a training phase to iteratively refine and generate the most suitable intent rubric for the LLM, with the goal of improving accuracy when evaluating intent ratings.
"""

TRAINING_INSTRUCTIONS = """
1. Input the base rubric prompt for extracting signals and generating schema. If left empty, a default prompt is used.

2. Input the intent rubric. If left empty, a default intent rubric is also used.
"""