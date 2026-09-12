# Topic interpretation notes

The topic labels used in the portfolio are **human interpretations**, not model-generated class names. They were assigned only after inspecting the automated top terms, document-assignment shares, and representative complaints produced by the pipeline. The raw representative complaint export is intentionally not committed to the public repository; it is recreated locally by `python run_analysis.py`.

## LDA topics

| Topic | Interpretive label | Evidence from highest-weight terms |
|---|---|---|
| 1 | Credit-file inaccuracies and mixed disputes | `credit`, `information`, `accounts`, `report`, `file`, `reporting`, `credit report`, `removed`, `credit file` |
| 2 | Account opening, balance, and charge details | `account`, `balance`, `opened`, `acct`, `date`, `charge`, `opened balance` |
| 3 | Identity theft and fraudulent accounts | `report`, `credit`, `accounts`, `identity`, `fraudulent`, `theft`, `identity theft`, `items`, `remove` |
| 4 | Debt collection and legal-compliance allegations | `claim`, `reporting`, `collection`, `debt`, `compliance`, `alleged`, `claims`, `plaintiff`, `document` |
| 5 | Formal inaccurate-information correction requests | `credit`, `information`, `report`, `inaccurate`, `inaccurate information`, `letter`, `following` |
| 6 | Payment, loan, and account-status reporting | `credit`, `account`, `report`, `reporting`, `late`, `accounts`, `payment`, `loan` |
| 7 | Identity-theft blocking and FCRA rights | `consumer`, `information`, `agency`, `reporting`, `section`, `block`, `consumer reporting`, `identity`, `theft` |
| 8 | Furnisher and inquiry evidence disputes | `information`, `credit`, `consumer`, `inquiry`, `furnisher`, `reporting`, `section`, `accounts`, `report` |

## LSA components

| Component | Interpretive label | Evidence from highest-weight terms |
|---|---|---|
| 1 | General credit-report and account disputes | `credit`, `report`, `account`, `accounts`, `credit report`, `information`, `reporting`, `fraudulent`, `identity`, `remove` |
| 2 | Fraudulent charge and account-opening details | `acct opened`, `acct`, `account acct`, `opened balance`, `balance account`, `charge`, `opened`, `balance` |
| 3 | Unknown or fraudulent credit-file items | `accounts`, `identity`, `items`, `theft`, `identity theft`, `pulled`, `fraudulent`, `unknown`, `pulled credit`, `remove` |
| 4 | Identity-theft victim and blocking-rights language | `theft`, `identity theft`, `identity`, `information`, `victim`, `victim identity`, `block`, `consumer`, `section` |

## Interpretation safeguards

- LDA document-topic weights are probabilities; the reported LDA shares use each document's highest topic probability.
- LSA coordinates are signed latent dimensions, not probabilities. LSA shares are therefore descriptive assignments based on the largest absolute component loading.
- The labels summarize recurring language patterns and do not constitute verified legal, factual, or demographic classifications of individual complaints.
- Exact duplicate narratives were removed, but near-duplicate legal or dispute templates remain a documented limitation and can influence learned topics.

The machine-generated terms supporting these interpretations are available in `outputs/tables/topic_terms.csv`, and the assignment shares are available in `outputs/tables/topic_prevalence.csv`.
