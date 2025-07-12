============================================================
      Overall Performance: Descriptive Statistics
============================================================
        recall  precision  clinical_reasoning  diagnostic_safety  \
mean     0.430      0.489               0.440              0.513   
median   0.500      0.464               0.486              0.500   
std      0.201      0.283               0.263              0.284   
min      0.000      0.000              -0.086              0.000   
max      0.833      1.000               0.857              1.000   

        clinical_coverage  
mean                0.783  
median              0.800  
std                 0.212  
min                 0.333  
max                 1.000  

Overall Mean Clinical Coverage: 78.33%

============================================================
                 Performance Per Case
============================================================
    case_id  recall  precision  clinical_reasoning  diagnostic_safety  clinical_coverage
0   Case 16   0.500      1.000               0.750              1.000              0.750
1   Case 21   0.167      0.200               0.089              0.200              0.333
2   Case 17   0.400      0.667               0.333              0.667              0.400
3   Case 11   0.333      0.500               0.714              0.500              1.000
4    Case 8   0.500      1.000               0.500              1.000              0.500
5   Case 19   0.000      0.000              -0.086              0.000              0.600
6   Case 22   0.500      0.333               0.471              0.383              1.000
7    Case 9   0.333      0.400               0.250              0.400              0.500
8    Case 3   0.667      0.571               0.657              0.657              1.000
9   Case 15   0.500      0.667               0.600              0.667              0.750
10  Case 13   0.167      0.167               0.175              0.167              0.833
11   Case 1   0.400      0.500               0.500              0.500              0.800
12   Case 2   0.250      0.250               0.120              0.250              0.750
13  Case 20   0.250      0.250               0.120              0.250              0.750
14   Case 5   0.667      0.400               0.333              0.400              0.667
15  Case 18   0.200      0.250               0.267              0.250              0.800
16  Case 10   0.600      0.500               0.600              0.600              1.000
17  Case 14   0.600      1.000               0.800              1.000              0.800
18   Case 4   0.833      0.833               0.857              0.833              1.000
19   Case 7   0.600      0.600               0.720              0.720              1.000
20   Case 6   0.500      0.429               0.575              0.514              1.000
21  Case 12   0.500      0.250               0.325              0.325              1.000

============================================================
          Fallback Agent Performance Analysis
============================================================
Top-performing agent was a 'Fallback' in 12 out of 22 runs (54.55%).

--- Mean Scores Comparison ---
                    Fallback Top Perf.  Custom Top Perf.
recall                           0.496             0.352
precision                        0.638             0.311
clinical_reasoning               0.530             0.331
diagnostic_safety                0.663             0.332
clinical_coverage                0.774             0.795
