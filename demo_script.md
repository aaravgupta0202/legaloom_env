# LegaLoom-Env: 90-Second Demo Script

## [0-15s] THE PROBLEM

"Every business in India deducts TDS before paying vendors. Get the section wrong — penalties. Miss an inoperative PAN — you owe 20% regardless. Current AI models get this wrong constantly."

Show: A sample invoice with PAN, amount, and service description.

## [15-35s] THE ENVIRONMENT

"We built LegaLoom-Env — the first OpenEnv-compliant RL environment for Indian TDS compliance. The agent reads a vendor invoice and must call a sequence of tools — read_invoice, check_pan, lookup_section — before submitting the exact TDS amount, section, and rate. No hints. Four difficulty levels from basic invoices to FY 2025-26 edge cases."

Show: Environment action space. The 4 difficulty levels.

## [35-55s] REWARD HACKING AUDIT

"Before training, we red-teamed our own reward function and found three exploits:
1. Hint leak — the environment whispered the next action. Patched.
2. Trainer impersonation — the reward function did the reasoning for the model. Patched.
3. Evidence-free no_tds — claim below-threshold without checking. Patched with −0.30 penalty."

Show: The Reward Hacking Audit section from README.

## [55-80s] TRAINING RESULTS

"We ran 40 GRPO steps with num_generations=8 on Qwen2.5-3B, focused on task_hard — inoperative-PAN scenarios, the most common TDS penalty trigger in real compliance. Hard improved +16%."

"What's interesting is the transfer effect. We trained only on hard, but every other task pool improved too — easy went up +20%, medium +8%, expert +4%. No regressions. The model didn't just memorize one rule; it learned a general compliance workflow that transfers across TDS section types."

Show: Before/after bar chart. Reward curves with phase labels.

## [80-90s] CLOSING

"LegaLoom-Env is the first RL environment for Indian TDS compliance. Deterministic rewards, OpenEnv-compliant, fully reproducible on Colab. The environment, training script, and all artifacts are on our HuggingFace Space."

END
