LegaLoom-Env: 90-Second Demo Script
=====================================

[0-15s] THE PROBLEM

"Every business in India deducts TDS before paying vendors. Get the section wrong — penalties. Miss an inoperative PAN — you owe 20% regardless. Current AI models get this wrong constantly."

Show: A sample invoice with PAN, amount, and service description.

[15-30s] BASELINE FAILURE

"Here's what an untrained LLM does. It reads the invoice, guesses the section, and submits — without checking if the PAN is operative."

Run: Baseline agent on task_hard (inoperative PAN).
Result: Agent submits 10% rate → Score: 0.12.
"Wrong. This PAN is inoperative. Section 206AA requires 20%."

[30-55s] TRAINED SUCCESS

"After GRPO training, the same model follows the correct reasoning chain."

Run: Trained agent on same invoice.
Step 1: read_invoice → sees invoice
Step 2: check_pan → "INOPERATIVE" flagged
Step 3: lookup_section → identifies correct section
Step 4: submit_answer → rate=20%, correct amount
Result: Score: 0.85.

"The trained agent checks PAN first, catches the inoperative flag, and applies the 20% override. That's the behavior we want."

[55-75s] TRICKY CASE: GST BUNDLING

"Here's a harder case — the invoice says 'inclusive of all taxes'. A naive agent deducts TDS on the pre-GST amount. The correct answer: when GST is bundled, TDS applies on the FULL invoice value."

Show: Reward curve comparison — baseline vs trained across 30 GRPO steps.

[75-90s] CLOSING

"LegaLoom-Env trains LLMs on real-world statutory compliance using GRPO with full episode rollouts. The environment is OpenEnv-compliant, deployable on HuggingFace Spaces, and the reward signal is deterministic — no room for reward hacking. Try it yourself."

END
