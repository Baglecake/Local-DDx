```text
Searching for JSON files in: /content/dual_llama3_transcripts
Successfully loaded and parsed: v7_3_duall_llama3_case_15_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_7_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_3_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_1_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_13_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_17_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_6_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_11_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_14_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_4_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_8_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_9_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_12_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_16_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_5_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_2_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_10_transcript.json

============================================================
      Overall Performance: Descriptive Statistics
============================================================
        recall  precision  clinical_reasoning  diagnostic_safety  \
mean     0.534      0.637               0.563              0.689   
median   0.500      0.667               0.650              0.767   
std      0.239      0.299               0.266              0.304   
min      0.000      0.000              -0.050              0.000   
max      1.000      1.000               1.000              1.000   

        clinical_coverage  
mean                0.813  
median              0.800  
std                 0.234  
min                 0.250  
max                 1.000  

Overall Mean Clinical Coverage: 81.27%

============================================================
                 Performance Per Case
============================================================
    case_id  recall  precision  clinical_reasoning  diagnostic_safety  clinical_coverage
0   Case 15   0.000      0.000              -0.050              0.000              0.250
1    Case 7   0.400      0.400               0.400              0.400              1.000
2    Case 3   0.167      0.200               0.200              0.200              1.000
3    Case 1   0.400      0.333               0.286              0.333              0.800
4   Case 13   0.833      0.833               0.950              0.950              1.000
5   Case 17   0.600      0.750               0.660              0.825              0.800
6    Case 6   0.667      0.667               0.767              0.767              1.000
7   Case 11   0.500      0.750               0.550              0.825              0.667
8   Case 14   1.000      1.000               1.000              1.000              1.000
9    Case 4   0.667      0.571               0.657              0.657              1.000
10   Case 8   0.500      0.667               0.575              0.767              0.750
11   Case 9   0.667      1.000               0.667              1.000              0.667
12  Case 12   0.500      0.500               0.650              0.650              1.000
13  Case 16   0.500      0.400               0.520              0.520              1.000
14   Case 5   0.333      1.000               0.333              1.000              0.333
15   Case 2   0.750      1.000               0.750              1.000              0.750
16  Case 10   0.600      0.750               0.660              0.825              0.800

============================================================
          Fallback Agent Performance Analysis
============================================================
Top-performing agent was a 'Fallback' in 0 out of 17 runs (0.00%).
Not enough data to compare fallback vs. custom top performers.
```
