This directory contains reports for version 8.5 of the ddx.

```markdown
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DDx BATCH ANALYSIS SUMMARY                           ║
║                            11 Cases Analyzed                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 AGGREGATE PERFORMANCE (Primary Clinical Metrics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clinical Success Rate (TP + AE):
   Mean:  66.2%   Median:  66.7%
   Range: 25.0% - 100.0%

Clinical Failure Rate (TM only):
   Mean:  15.5%   Median:   0.0%
   Range:  0.0% - 75.0%

Diagnostic Precision:
   Mean:  52.7%   Median:  50.0%
   Range: 25.0% - 100.0%

Traditional Recall (For Comparison):
   Mean:  39.7%   Median:  33.3%
   Range: 16.7% - 66.7%

🏆 PERFORMANCE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   EXCELLENT :   3 cases ( 27.3%)
   GOOD      :   5 cases ( 45.5%)
   FAIR      :   2 cases ( 18.2%)
   POOR      :   1 cases (  9.1%)

📈 KEY INSIGHTS (Clinical Binary Perspective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Clinical Success vs Traditional Recall gap: +26.5%
  This gap represents the clinical value of appropriate diagnostic exclusions.

• Average Clinical Performance: 66.2% success rate
  This credits both exact matches AND appropriate exclusions with reasoning.

• Average Clinical Failure Rate: 15.5%
  This represents only true diagnostic inadequacy (complete misses).

• Best performing case: Case 3
  (100.0% clinical success)

• Most challenging case: Case 8
  (25.0% clinical success)

CLINICAL INSIGHT: The normalized binary approach shows that systems perform better
clinically than traditional metrics suggest - Clinical reasoning includes
both correct inclusions AND appropriate exclusions with evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
