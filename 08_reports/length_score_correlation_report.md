# Text Length vs Score Correlation Report

Purpose: verify the model is NOT scoring longer texts higher (anti-shortcut test).
Pass condition: |Pearson r| < 0.3 (weak or no correlation between text length and score).

## MDA
- documents: 149
- word count: mean=3123.3, median=2047, range=[291, 15423]
- score: mean=49.96, median=46.25, range=[23.75, 93.75]
- **Pearson(word, score): r=-0.0424**
- **Spearman(word, score): r=0.1305**
- Pearson(char, score): r=-0.0443
- anti_shortcut_pass: **True**
- top 5 longest mean score: 45.75
- top 5 shortest mean score: 33.25

## News
- documents: 125
- word count: mean=394.2, median=369, range=[72, 961]
- score: mean=94.42, median=100.0, range=[40.0, 100.0]
- **Pearson(word, score): r=-0.172**
- **Spearman(word, score): r=-0.2918**
- Pearson(char, score): r=-0.1432
- anti_shortcut_pass: **True**
- top 5 longest mean score: 96.5
- top 5 shortest mean score: 100.0

## Interpretation
- |r| < 0.3: No meaningful length bias (passed)
- 0.3 <= |r| < 0.5: Weak length bias (needs discussion)
- |r| >= 0.5: Moderate/strong length bias (problematic for validity)