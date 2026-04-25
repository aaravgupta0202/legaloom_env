LegaLoom-Env: 90-Second Demo Script
=====================================

[0-15s] THE PROBLEM

"Every business in India deducts TDS before paying vendors. Get the section wrong — penalties. Miss an inoperative PAN — you owe 20% regardless. Current AI models get this wrong constantly."

Show: A sample invoice with PAN, amount, and service description.

[15-35s] THE ENVIRONMENT

"We built LegaLoom-Env — an OpenEnv-compliant environment that simulates the TDS compliance back-office. The agent reads a vendor invoice and must call a sequence of tools — read_invoice, check_pan, lookup_section — before submitting the exact TDS amount, section, and rate. No hints. Four difficulty levels."

Show: Environment action space table. Highlight the 4-phase curriculum (easy → medium → hard → expert).

[35-55s] REWARD HACKING AUDIT

"Before training, we red-teamed our own reward function and found three exploits:
1. Hint leak — the environment was whispering the next correct tool call. Patched.
2. Trainer impersonation — the reward function was injecting actions on behalf of the model. Patched.
3. Evidence-free no_tds claim — the model could claim below-threshold without evidence. Patched with a −0.30 penalty."

Show: The Reward Hacking Audit section from README.

[55-75s] TRAINING RESULTS

"We ran 80 GRPO steps across all four tasks using a 4-phase curriculum on Qwen2.5-3B. Medium tasks cross the success threshold at 0.528 and remain stable. The reward curves show real phase-aware dynamics — you can see the curriculum in action."

Show: Reward curves with phase boundaries. Before/after bar chart. Highlight that medium has NO regression (unlike earlier 2-phase training).

[75-90s] CLOSING

"LegaLoom-Env is the first RL environment for Indian TDS compliance. The reward signal is deterministic, the environment is OpenEnv-compliant, and the notebook is fully reproducible on HuggingFace Spaces. Try it yourself."

END
