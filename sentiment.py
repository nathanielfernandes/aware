import os
from time import perf_counter

print("Loading sentiment classifier... ⌛")

tick = perf_counter()

os.environ["TRANSFORMERS_CACHE"] = "models"
from transformers import pipeline

# Create the pipeline
analyze_sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)

tock = perf_counter()
print(f"Loaded sentiment classifier in {tock - tick:.4f}s ✅")


def analyze(text: str) -> dict:
    try:
        tick = perf_counter()
        # Analyze the sentiment of the text
        result = analyze_sentiment(text, top_k=3)
        tock = perf_counter()

        positive = next(filter(lambda x: x["label"] == "positive", result))["score"]
        negative = next(filter(lambda x: x["label"] == "negative", result))["score"]
        neutral = next(filter(lambda x: x["label"] == "neutral", result))["score"]

        # ms
        print(f"Analyzed sentiment in {round((tock - tick) * 1000)}ms 🔥")

        # Return the result
        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        }
    except Exception as e:
        print(f"Error analyzing sentiment: {e} ⭕")
        # Return an error
        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
        }
