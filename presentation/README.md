# Presentation and historical results

The original classroom deck was located in Downloads as `Final_Group_5.pptx`. Its title slide identifies Patient Drug Event Analysis, Group 5, Uma Viraja Dasaraju and Shreya Saraf. A copy is included here as `Patient_Drug_Event_Analysis_Presentation.pptx`. The original presentation is preserved without edits.

The guide's historical reproducible rerun reports the following values. These are reference figures, not measurements from this build:

| Model | Baseline accuracy | Baseline death recall | Baseline death F1 | SMOTENC accuracy | SMOTENC death recall | SMOTENC death F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 96.41% | 58.48% | 70.65% | 90.73% | 69.82% | 52.72% |
| Random Forest | 94.29% | 22.79% | 37.12% | 92.27% | 63.51% | 54.86% |
| Linear SVM | 96.35% | 58.79% | 70.40% | 89.47% | 70.72% | 49.84% |
| KNN | 95.11% | 45.37% | 57.82% | 71.37% | 79.73% | 29.18% |

The guide separately describes classroom baseline accuracy around 64–67% and SMOTENC accuracy around 47–51%. Do not combine those with this build's results. Training-only rare-category grouping and scaling before SMOTENC are intentional methodological corrections in this build, so exact guide metrics are not expected to match.
