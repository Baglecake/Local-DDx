This directory contains reports for version 8.5 of the ddx.

```markdown
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DDx v8.5 BATCH ANALYSIS SUMMARY                       ║
║                            12 Cases Analyzed                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 AGGREGATE PERFORMANCE (Primary Clinical Metrics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clinical Success Rate (TP + AE):
   Mean:  69.0%   Median:  66.7%
   Range: 25.0% - 100.0%

Clinical Failure Rate (TM only):
   Mean:  14.2%   Median:   0.0%
   Range:  0.0% - 75.0%

Diagnostic Precision:
   Mean:  51.1%   Median:  41.7%
   Range: 25.0% - 100.0%

Traditional Recall (For Comparison):
   Mean:  44.7%   Median:  36.7%
   Range: 16.7% - 100.0%

🏆 PERFORMANCE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   EXCELLENT :   4 cases ( 33.3%)
   GOOD      :   5 cases ( 41.7%)
   FAIR      :   2 cases ( 16.7%)
   POOR      :   1 cases (  8.3%)

📈 KEY INSIGHTS (Clinical Binary Perspective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Clinical Success vs Traditional Recall gap: +24.3%
  This gap represents the clinical value of appropriate diagnostic exclusions.

• Average Clinical Performance: 69.0% success rate
  This credits both exact matches AND appropriate exclusions with reasoning.

• Average Clinical Failure Rate: 14.2%
  This represents only true diagnostic inadequacy (complete misses).

• Best performing case: Case 3
  (100.0% clinical success)

• Most challenging case: Case 8
  (25.0% clinical success)

CLINICAL INSIGHT: The normalized binary approach shows that systems perform better
clinically than traditional metrics suggest, because good clinical reasoning includes
both correct inclusions AND appropriate exclusions with evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
