# Data

The raw file is **not committed** (about 100 MB). Fetch it with:

```bash
make data                                  # mirror, no account needed
python -m fraudlab download --source kaggle  # official source, needs the kaggle CLI + API token
```

**Dataset:** Credit Card Fraud Detection, Worldline and the Machine Learning Group of
Université Libre de Bruxelles (ULB), released on Kaggle under the Database Contents License
(DbCL) v1.0. It holds 284,807 European card transactions over two days in September 2013,
of which 492 are fraudulent (0.172%).

| Column | Meaning |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1`–`V28` | Principal components of the original (confidential) features |
| `Amount` | Transaction amount |
| `Class` | 1 = fraud, 0 = legitimate |

Because `V1`–`V28` are PCA outputs, their business meaning is unknown, and only `Time` and
`Amount` can be feature-engineered.

**Citation:** Dal Pozzolo, A., Caelen, O., Johnson, R. A., Bontempi, G. *Calibrating Probability
with Undersampling for Unbalanced Classification.* IEEE CIDM, 2015.

`fraudlab download` verifies the SHA-256 of the file against the copy this project was developed
with and warns if it differs.
