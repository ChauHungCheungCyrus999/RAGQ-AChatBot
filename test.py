import re

# ====== Utility to Clean Model Output ======
def clean_response(response_text):
    # Remove <think>...</think> blocks (internal reasoning from model, if leaked)
    no_think = re.sub(r"<think>.*?</think>\n?", "", response_text, flags=re.DOTALL)
    # Remove everything up to and including the first "A:" marker (case-insensitive)
    cleaned = re.sub(r'^.*?A:\s*', '', no_think, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

# ====== Test clean_response function ======
def test_clean_response():
    test_cases = [
        "A: i love apple.",
        "Some intro text A: Here is the answer.",
        "<think>This is internal logic</think>\nA: Clean output here.",
        "<think>Reasoning stuff</think>\nRandom text before A: Final answer.",
        "a: lowercase A marker should also be removed.",
        "No marker here, just some text.",
        "<think>Some thought</think> No A marker here either.",
    ]

    for i, text in enumerate(test_cases):
        cleaned = clean_response(text)
        print(f"Test case {i+1}:")
        print(f"Original: {repr(text)}")
        print(f"Cleaned:  {repr(cleaned)}\n")

# Run the test
test_clean_response()