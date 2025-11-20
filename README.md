Rubric Overview
| #     | Criterion                                         | Description                                                                                        | Score Scale                                                                                                                                   | Weight   | Comment |
| ----- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------
| **1** | **Data Accessibility & Representativeness**       | Availability of data, clarity on patient demographics, scanner info, inclusion/exclusion criteria. | **0**: Private data only<br>**1**: Partially public or “available upon request”<br>**2**: Fully public dataset or clearly reproducible cohort | **2×**   ||
| **2** | **Annotation Protocol & Inter-Rater Reliability** | Detail on annotation procedure, rater expertise, agreement metrics.                                | **0**: No info<br>**1**: Described but no agreement metrics<br>**2**: Full protocol with inter-rater reliability                              | **1.5×** | If the paper uses a public dataset without creating new annotations, give it a full score. If the authors create new annotations, evaluate criterion normally.|
| **3** | **Code Availability & Executability**             | Access to runnable code and clear instructions.                                                    | **0**: No code<br>**1**: Partial scripts<br>**2**: Full runnable repository (Docker/Colab etc.)                                               | **2×**   |
| **4** | **Preprocessing & Normalization Transparency**    | Clarity of preprocessing steps (resampling, cropping, normalization, augmentation).                | **0**: Unspecified<br>**1**: Partial description<br>**2**: Fully documented and reproducible                                                  | **1.5×** |
| **5** | **Evaluation Protocol & Statistical Reporting**   | Train/test splits, cross-validation, metrics, uncertainty reporting.                               | **0**: Unclear<br>**1**: Partial<br>**2**: Fully described with confidence intervals or multiple runs                                         | **1.5×** |
| **6** | **Documentation & Instructions**                  | Ease of running the code from repository instructions.                                             | **0**: None<br>**1**: Minimal<br>**2**: Clear and complete                                                                                    | **1×**   |
| **7** | **Licensing, Ethics & Governance**                | Licensing and ethical transparency (IRB, consent, anonymization).                                  | **0**: Missing<br>**1**: Partial<br>**2**: Open license + ethics/consent statement                                                            | **1×**   |


Step 2. Download PDFs
Step 3. Extract Text
Step 4. Regex-Based Scoring (Tier 1)
Code Availability (0–2)
Data Availability (0–2)
Ethics / Governance (0–2)
Step 5. Combine Scores
Step 6. LLM-Assisted Scoring (Tier 2)
