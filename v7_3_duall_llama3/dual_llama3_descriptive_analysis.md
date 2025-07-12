```text
Searching for JSON files in: /content/dual_llama3_transcripts
Successfully loaded and parsed: v7_3_duall_llama3_case_15_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_7_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_3_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_18_transcript.json
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
Successfully loaded and parsed: v7_3_duall_llama3_case_19_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_5_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_2_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_10_transcript.json
Successfully loaded and parsed: v7_3_duall_llama3_case_20_transcript.json

============================================================
      Overall Performance: Descriptive Statistics
============================================================
        recall  precision  clinical_reasoning  diagnostic_safety  \
mean     0.519      0.614               0.542              0.664   
median   0.500      0.667               0.575              0.767   
std      0.223      0.284               0.254              0.294   
min      0.000      0.000              -0.050              0.000   
max      1.000      1.000               1.000              1.000   

        clinical_coverage  
mean                0.808  
median              0.800  
std                 0.225  
min                 0.250  
max                 1.000  

Overall Mean Clinical Coverage: 80.83%

============================================================
                 Performance Per Case
============================================================
    case_id  recall  precision  clinical_reasoning  diagnostic_safety  clinical_coverage
0   Case 15   0.000      0.000              -0.050              0.000              0.250
1    Case 7   0.400      0.400               0.400              0.400              1.000
2    Case 3   0.167      0.200               0.200              0.200              1.000
3   Case 18   0.400      0.400               0.400              0.400              1.000
4    Case 1   0.400      0.333               0.286              0.333              0.800
5   Case 13   0.833      0.833               0.950              0.950              1.000
6   Case 17   0.600      0.750               0.660              0.825              0.800
7    Case 6   0.667      0.667               0.767              0.767              1.000
8   Case 11   0.500      0.750               0.550              0.825              0.667
9   Case 14   1.000      1.000               1.000              1.000              1.000
10   Case 4   0.667      0.571               0.657              0.657              1.000
11   Case 8   0.500      0.667               0.575              0.767              0.750
12   Case 9   0.667      1.000               0.667              1.000              0.667
13  Case 12   0.500      0.500               0.650              0.650              1.000
14  Case 16   0.500      0.400               0.520              0.520              1.000
15  Case 19   0.400      0.400               0.286              0.400              0.600
16   Case 5   0.333      1.000               0.333              1.000              0.333
17   Case 2   0.750      1.000               0.750              1.000              0.750
18  Case 10   0.600      0.750               0.660              0.825              0.800
19  Case 20   0.500      0.667               0.575              0.767              0.750

============================================================
          Fallback Agent Performance Analysis
============================================================
Top-performing agent was a 'Fallback' in 0 out of 20 runs (0.00%).
Not enough data to compare fallback vs. custom top performers.
```
