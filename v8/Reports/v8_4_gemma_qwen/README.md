Case reports from 30 runs of the gemma2 qwen2.5 setup.
```markdown
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DDx BATCH ANALYSIS SUMMARY                           ║
║                            30 Cases Analyzed                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 AGGREGATE PERFORMANCE (Primary Clinical Metrics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clinical Success Rate (TP + AE):
   Mean:  63.0%   Median:  66.7%
   Range: 14.3% - 100.0%

Clinical Failure Rate (TM only):
   Mean:  22.2%   Median:   8.3%
   Range:  0.0% - 80.0%

Diagnostic Precision:
   Mean:  46.8%   Median:  40.0%
   Range:  0.0% - 100.0%

Traditional Recall (For Comparison):
   Mean:  38.6%   Median:  40.0%
   Range:  0.0% - 100.0%

🏆 PERFORMANCE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   EXCELLENT :   8 cases ( 26.7%)
   GOOD      :  10 cases ( 33.3%)
   FAIR      :   6 cases ( 20.0%)
   POOR      :   6 cases ( 20.0%)

📈 KEY INSIGHTS (Clinical Binary Perspective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Clinical Success vs Traditional Recall gap: +24.4%
  This gap represents the clinical value of appropriate diagnostic exclusions.

• Average Clinical Performance: 63.0% success rate
  This credits both exact matches AND appropriate exclusions with reasoning.

• Average Clinical Failure Rate: 22.2%
  This represents only true diagnostic inadequacy (complete misses).

• Best performing case: Case 12
  (100.0% clinical success)

• Most challenging case: Case 29
  (14.3% clinical success)

CLINICAL INSIGHT: The normalized binary approach shows that systems perform better
clinically than traditional metrics suggest, because good clinical reasoning includes
both correct inclusions AND appropriate exclusions with evidence.
```
