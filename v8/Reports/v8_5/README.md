This directory contains reports for version 8.5 of the ddx.

```markdown
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DDx v8.5 BATCH ANALYSIS SUMMARY                       ║
║                            15 Cases Analyzed                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 AGGREGATE PERFORMANCE (Primary Clinical Metrics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clinical Success Rate (TP + AE):
   Mean:  69.1%   Median:  66.7%
   Range: 25.0% - 100.0%

Clinical Failure Rate (TM only):
   Mean:  13.4%   Median:   0.0%
   Range:  0.0% - 75.0%

Diagnostic Precision:
   Mean:  52.5%   Median:  50.0%
   Range: 25.0% - 100.0%

Traditional Recall (For Comparison):
   Mean:  45.9%   Median:  40.0%
   Range: 16.7% - 100.0%

🏆 PERFORMANCE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   EXCELLENT :   4 cases ( 25.0%)
   GOOD      :   9 cases ( 56.2%)
   FAIR      :   2 cases ( 12.5%)
   POOR      :   1 cases (  6.2%)

📈 KEY INSIGHTS (Clinical Binary Perspective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Clinical Success vs Traditional Recall gap: +23.1%
  This gap represents the clinical value of appropriate diagnostic exclusions.

• Average Clinical Performance: 69.1% success rate
  This credits both exact matches AND appropriate exclusions with reasoning.

• Average Clinical Failure Rate: 13.4%
  This represents only true diagnostic inadequacy (complete misses).

• Best performing case: Case 3
  (100.0% clinical success)

• Most challenging case: Case 8
  (25.0% clinical success)

CLINICAL INSIGHT: The normalized binary approach shows that systems perform better
clinically than traditional metrics suggest- clinical reasoning includes
both correct inclusions AND appropriate exclusions with evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
