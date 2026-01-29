import os
import json
import logging
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Ensure the API key is set
api_key = os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    logging.error("GPT_API_KEY is not set in the environment variables.")
    raise ValueError(
        "API key not found. Please set the GPT_API_KEY environment variable."
    )

client = OpenAI(api_key=api_key)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compare-words", methods=["POST"])
def compare_words():
    data = request.get_json()
    word1 = data.get("word1", "").strip().lower()
    word2 = data.get("word2", "").strip().lower()

    if not word1 or not word2:
        return jsonify({"error": "Please provide both words to compare."}), 400

    # Explore both words individually (STRICT QUOTE RULES)
    word1_prompt = f"""Explore the word "{word1}" concisely. Return a JSON object:
{{
    "etymology": "Brief origin (1 sentence)",
    "synonyms": ["4-5 synonyms"],
    "metaphors": ["2 common metaphors"],
    "quotes": [
        "1-2 REAL, verifiable quotes containing this word, with author/source.
         If none exist, return an empty array."
    ]
}}
IMPORTANT:
- Do NOT invent quotes.
- Do NOT use anonymous or fake authors.
Return ONLY valid JSON."""

    word2_prompt = f"""Explore the word "{word2}" concisely. Return a JSON object:
{{
    "etymology": "Brief origin (1 sentence)",
    "synonyms": ["4-5 synonyms"],
    "metaphors": ["2 common metaphors"],
    "quotes": [
        "1-2 REAL, verifiable quotes containing this word, with author/source.
         If none exist, return an empty array."
    ]
}}
IMPORTANT:
- Do NOT invent quotes.
- Do NOT use anonymous or fake authors.
Return ONLY valid JSON."""

    # Bridge prompt (STRICT QUOTE RULES)
    bridge_prompt = f"""Find connections between "{word1}" and "{word2}". Return a JSON object:
{{
    "quotes_with_both": [
        "2-3 REAL, verifiable literary quotes or famous sayings containing BOTH words, including author/source.
         If none exist, return an empty array."
    ],
    "shared_idioms": [
        "2-3 idioms or phrases that pair these words together, if any exist.
         If none exist, return an empty array."
    ],
    "conceptual_links": ["3-4 ways these words connect thematically or philosophically"],
    "shared_metaphors": ["2-3 metaphors that could apply to both words"],
    "tension": "One sentence describing the creative tension or contrast between these words"
}}

IMPORTANT RULES:
- Do NOT fabricate quotes.
- Do NOT create fake attributions.
- If no verified quote exists, return [].

Return ONLY valid JSON."""

    try:
        # Make calls for each prompt
        responses = []
        for prompt in [word1_prompt, word2_prompt, bridge_prompt]:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a linguistic and literary expert. "
                            "Always respond with valid JSON only. "
                            "Never invent quotations or sources."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.7,
            )
            responses.append(response.choices[0].message.content.strip())

        results = []
        for text_response in responses:
            # Strip markdown fences if present
            if text_response.startswith("```"):
                lines = text_response.split("\n")
                text_response = "\n".join(lines[1:-1])

            results.append(json.loads(text_response))

        return jsonify(
            {
                "word1": word1,
                "word2": word2,
                "constellation1": results[0],
                "constellation2": results[1],
                "bridge": results[2],
            }
        )

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response: {e}")
        return (
            jsonify({"error": "Failed to process the comparison. Please try again."}),
            500,
        )

    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return jsonify({"error": "Failed to compare words. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
