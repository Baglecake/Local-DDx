```text
============================================================
      Overall Performance: Descriptive Statistics
============================================================
        recall  precision  clinical_reasoning  diagnostic_safety  \
mean     0.452      0.491               0.448              0.514   
median   0.500      0.500               0.500              0.500   
std      0.213      0.273               0.256              0.275   
min      0.000      0.000              -0.086              0.000   
max      1.000      1.000               0.857              1.000   

        clinical_coverage  
mean                0.804  
median              0.800  
std                 0.204  
min                 0.333  
max                 1.000  

Overall Mean Clinical Coverage: 80.37%

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
12  Case 27   0.500      0.500               0.657              0.600              1.000
13  Case 15   0.500      0.667               0.600              0.667              0.750
14  Case 23   0.333      0.200               0.200              0.200              1.000
15  Case 13   0.167      0.167               0.175              0.167              0.833
16   Case 1   0.400      0.500               0.500              0.500              0.800
17   Case 2   0.250      0.250               0.120              0.250              0.750
18  Case 20   0.250      0.250               0.120              0.250              0.750
19   Case 5   0.667      0.400               0.333              0.400              0.667
20  Case 18   0.200      0.250               0.267              0.250              0.800
21  Case 10   0.600      0.500               0.600              0.600              1.000
22  Case 14   0.600      1.000               0.800              1.000              0.800
23   Case 4   0.833      0.833               0.857              0.833              1.000
24   Case 7   0.600      0.600               0.720              0.720              1.000
25   Case 6   0.500      0.429               0.575              0.514              1.000
26  Case 12   0.500      0.250               0.325              0.325              1.000

============================================================
          Fallback Agent Performance Analysis
============================================================
Top-performing agent was a 'Fallback' in 15 out of 27 runs (55.56%).

--- Mean Scores Comparison ---
                    Fallback Top Perf.  Custom Top Perf.
recall                           0.486             0.410
precision                        0.607             0.346
clinical_reasoning               0.519             0.359
diagnostic_safety                0.634             0.363
clinical_coverage                0.797             0.812
```
<img width="1184" height="684" alt="image" src="https://github.com/user-attachments/assets/c57d688c-afd4-484c-bf75-7269a67f784c" />

<img width="1584" height="784" alt="image" src="https://github.com/user-attachments/assets/978ca529-6e53-4d41-8db1-34fdc1954135" />

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/ea3f18d8-3c38-41d4-b6c5-ce0e77bb94bc" />
