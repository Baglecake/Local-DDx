This directory contains reports for version 8.5 of the ddx.

```markdown
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DDx v8.5 BATCH ANALYSIS SUMMARY                       ║
║                            14 Cases Analyzed                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 AGGREGATE PERFORMANCE (Primary Clinical Metrics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clinical Success Rate (TP + AE):
   Mean:  68.2%   Median:  66.7%
   Range: 25.0% - 100.0%

Clinical Failure Rate (TM only):
   Mean:  13.6%   Median:   0.0%
   Range:  0.0% - 75.0%

Diagnostic Precision:
   Mean:  52.1%   Median:  50.0%
   Range: 25.0% - 100.0%

Traditional Recall (For Comparison):
   Mean:  43.6%   Median:  36.7%
   Range: 16.7% - 100.0%

🏆 PERFORMANCE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   EXCELLENT :   4 cases ( 28.6%)
   GOOD      :   7 cases ( 50.0%)
   FAIR      :   2 cases ( 14.3%)
   POOR      :   1 cases (  7.1%)

📈 KEY INSIGHTS (Clinical Binary Perspective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Clinical Success vs Traditional Recall gap: +24.6%
  This gap represents the clinical value of appropriate diagnostic exclusions.

• Average Clinical Performance: 68.2% success rate
  This credits both exact matches AND appropriate exclusions with reasoning.

• Average Clinical Failure Rate: 13.6%
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
