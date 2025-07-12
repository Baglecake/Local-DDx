```text
Searching for JSON files in: /content/transcripts
Successfully loaded and parsed: ddx_v7_2_case_16_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_25_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_26_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_21_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_17_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_11_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_8_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_19_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_24_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_22_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_9_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_3_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_15_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_23_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_13_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_1_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_2_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_20_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_5_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_18_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_10_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_14_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_4_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_7_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_6_transcript.json
Successfully loaded and parsed: ddx_v7_2_case_12_transcript.json

============================================================
      Overall Performance: Descriptive Statistics
============================================================
        recall  precision  clinical_reasoning  diagnostic_safety  \
mean     0.450      0.490               0.440               0.51   
median   0.500      0.464               0.486               0.50   
std      0.217      0.279               0.258               0.28   
min      0.000      0.000              -0.086               0.00   
max      1.000      1.000               0.857               1.00   

        clinical_coverage  
mean                0.796  
median              0.800  
std                 0.204  
min                 0.333  
max                 1.000  

Overall Mean Clinical Coverage: 79.62%

============================================================
                 Performance Per Case
============================================================
    case_id  recall  precision  clinical_reasoning  diagnostic_safety  clinical_coverage
0   Case 16   0.500      1.000               0.750              1.000              0.750
1   Case 25   1.000      0.750               0.750              0.750              1.000
2   Case 26   0.500      0.750               0.571              0.750              0.667
3   Case 21   0.167      0.200               0.089              0.200              0.333
4   Case 17   0.400      0.667               0.333              0.667              0.400
5   Case 11   0.333      0.500               0.714              0.500              1.000
6    Case 8   0.500      1.000               0.500              1.000              0.500
7   Case 19   0.000      0.000              -0.086              0.000              0.600
8   Case 24   0.400      0.286               0.250              0.286              0.800
9   Case 22   0.500      0.333               0.471              0.383              1.000
10   Case 9   0.333      0.400               0.250              0.400              0.500
11   Case 3   0.667      0.571               0.657              0.657              1.000
12  Case 15   0.500      0.667               0.600              0.667              0.750
13  Case 23   0.333      0.200               0.200              0.200              1.000
14  Case 13   0.167      0.167               0.175              0.167              0.833
15   Case 1   0.400      0.500               0.500              0.500              0.800
16   Case 2   0.250      0.250               0.120              0.250              0.750
17  Case 20   0.250      0.250               0.120              0.250              0.750
18   Case 5   0.667      0.400               0.333              0.400              0.667
19  Case 18   0.200      0.250               0.267              0.250              0.800
20  Case 10   0.600      0.500               0.600              0.600              1.000
21  Case 14   0.600      1.000               0.800              1.000              0.800
22   Case 4   0.833      0.833               0.857              0.833              1.000
23   Case 7   0.600      0.600               0.720              0.720              1.000
24   Case 6   0.500      0.429               0.575              0.514              1.000
25  Case 12   0.500      0.250               0.325              0.325              1.000

============================================================
          Fallback Agent Performance Analysis
============================================================
Top-performing agent was a 'Fallback' in 14 out of 26 runs (53.85%).

--- Mean Scores Comparison ---
                    Fallback Top Perf.  Custom Top Perf.
recall                           0.485             0.410
precision                        0.615             0.346
clinical_reasoning               0.510             0.359
diagnostic_safety                0.636             0.363
clinical_coverage                0.782             0.812
```
