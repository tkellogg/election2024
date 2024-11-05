# election2024

For the US Presidential Election 2024, I wrote the script to help me do candidate research. There's 20+ things to vote on, and I want to do my best
to make an informed decision, but that's a lot of work.

## Run it

1. Clone the repo
2. Create a `.env` file with `ANTHROPIC_API_KEY=` set
3. Edit candidate_recommender.py line ~153 or so, for `preferences`, describe what party you normally vote for, issues you care about, reasons you often switch sides, etc.
4. Run `rye run python candidate_recommender.py`

When it runs, you select one race at a time to analyze & give recommendations for

How does it know about the races? JSON. I sent pictures of a sample ballot to Claude and it spit out JSON describing the races. The script picks up all `*.json` files, they
just have to match the schema.

## How does it work?
It's a 3 step process:

1. For each candidate, search duckduckgo 
2. Use Claude to summarize the critical issues
3. Use Claude to make a recommendation + rationale, based on my preferences.

For each of these, I use [DSPy](https://dspy-docs.vercel.app/intro/) to wrangle the LLMs, although, I didn't bother with any DSPy prompt optimization.
