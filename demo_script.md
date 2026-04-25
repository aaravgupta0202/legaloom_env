LegaLoom-Env: 90-Second Demo Script
=====================================

[0-15s] THE PROBLEM

"Every business in India deducts TDS before paying vendors. Get the section wrong — penalties. Miss an inoperative PAN — you owe 20% regardless. Current AI models get this wrong constantly."

Show: A sample invoice with PAN, amount, and service description.

[15-30s] BASELINE FAILURE

"Here's what an untrained Qwen2.5-3B does on the inoperative-PAN scenarios in `task_hard`. It reads the invoice, guesses the section, and submits — without checking if the PAN is operative."

Run: Baseline agent (untrained Qwen2.5-3B-Instruct) on `task_hard`.
Result over 5 episodes: average score **0.078** — well below the 0.5 success threshold.
"Wrong. The model never applies the 20% Section 206AA override."

[30-55s] TRAINED RESPONSE

"After 40 GRPO steps (20 on easy, 20 on hard) using full episode rollouts, the same model improves on three of four tasks."

Run: Trained agent on `task_hard`.
Step 1: read_invoice → sees invoice
Step 2: check_pan → "INOPERATIVE" flagged
Step 3: lookup_section → identifies correct section
Step 4: submit_answer with rate=20%

Result over 5 episodes: average score **0.126** — a **+62% relative gain** on the hardest task in the suite. Not above the success threshold yet at this compute budget, but the direction is correct and reproducible.

[55-75s] HONEST ABOUT THE TRADE-OFF

"One task got worse: `task_medium` dropped from 0.450 to 0.336. Medium wasn't in the training curriculum — the inoperative-PAN signal we taught the model on `task_hard` over-triggers on threshold-boundary scenarios. We're showing this regression instead of hiding it. With more compute we'd interleave medium into the curriculum."

Show: Before/after bar chart with all 4 tasks; reward curves from `training_log.json`.

[75-90s] CLOSING

"LegaLoom-Env trains LLMs on real-world statutory compliance using GRPO with full-episode rollouts. The environment is OpenEnv-compliant, deployable on HuggingFace Spaces, and we red-teamed our own reward function before training — three exploits found and patched, all documented. Try it yourself."

END
