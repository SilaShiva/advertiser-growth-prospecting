# Data-Quality Gate Report

**Blocked:** no

| Gate | Severity | Passed | Detail |
|---|---|---|---|
| row_count_min | FAIL | PASS | 842 rows (min 700) |
| id_uniqueness | FAIL | PASS | 0 duplicate (brand, quarter) keys |
| label_parse_rate | FAIL | PASS | 1.0000 parseable labels (min 0.995) |
| label_values | FAIL | PASS | labels in {0,1} |
| null_share_numeric | WARN | PASS | worst null share 0.058 in 'app_installs_est' (max 0.15) |
| category_cardinality | FAIL | PASS | 8 categories (expected 8) |
| spend_nonnegative | FAIL | PASS | 0 negative spend rows |
| duplicate_rows | WARN | PASS | 0 exact duplicate rows (tolerated 5) |