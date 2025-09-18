PROJECT_ID = "mechanigo-liveagent"
DATASET_NAME = "conversations"

# ========== Feedback Loop ========== 
MAX_ITER = 3
CONSECUTIVE_NO_IMPROVEMENT = 0 # counter for iterations that have no improvement in accuracy
MAX_CONSECUTIVE_NO_IMPROVEMENT = 2 # One less than the max iteration you set
TEMPERATURE = 0.1
MAX_CONCURRENT = 10

# Meta commentaries ni ChatGPT that we want to remove sa mga rubric
# You may add more patterns here
UNWANTED_PATTERNS = [
    r"The rubric aims.*",
    r"This rubric aims.*",
    r"The goal of this rubric.*",
    r"This evaluation rubric.*",
    r"In summary, this rubric.*"
]