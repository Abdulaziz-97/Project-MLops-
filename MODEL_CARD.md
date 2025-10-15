# Model Card - Census Income Prediction

## Model Details

**Model Type:** Random Forest Classifier  
**Framework:** scikit-learn (version 1.3.0+)  
**Model Version:** 1.0.0  
**Date Created:** October 2025  
**Developers:** Udacity MLOps Nanodegree Project  
**License:** MIT  

This model is a supervised machine learning classifier that predicts whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from the 1994 US Census dataset.

**Model Architecture:**
- Algorithm: RandomForestClassifier
- Number of estimators: 100 trees
- Random state: 42 (for reproducibility)
- Default scikit-learn hyperparameters for all other settings

## Intended Use

**Primary Intended Uses:**
- Educational demonstration of MLOps practices and model deployment
- Learning exercise for end-to-end machine learning pipelines
- Understanding income prediction based on demographic factors
- Analyzing model performance across demographic slices

**Intended Users:**
- Data science students and ML practitioners
- Researchers studying income inequality
- Educational institutions teaching ML/MLOps
- Developers learning API deployment

**Out-of-Scope Uses:**
- **NOT** for making actual hiring, lending, or financial decisions
- **NOT** for determining individual creditworthiness
- **NOT** for any real-world discrimination or classification of individuals
- **NOT** for production use without extensive additional validation
- **NOT** for making decisions that could impact people's lives or livelihoods

## Training Data

**Dataset:** UCI Adult Census Income Dataset  
**Source:** https://archive.ics.uci.edu/ml/datasets/census+income  
**Original Size:** 32,561 samples with 15 features  
**Train/Test Split:** 80/20 (26,048 training samples, 6,513 test samples)  
**Random Seed:** 42 (for reproducibility)

**Class Distribution:**
- Income ≤$50K: ~76% (majority class)
- Income >$50K: ~24% (minority class)

**Features Used:**

*Demographic Features:*
- `age`: Integer, age in years
- `race`: Categorical (White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other)
- `sex`: Binary (Male, Female)
- `native-country`: Categorical, country of origin

*Employment Features:*
- `workclass`: Categorical (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, etc.)
- `occupation`: Categorical (Exec-managerial, Prof-specialty, Sales, Craft-repair, etc.)
- `hours-per-week`: Integer, hours worked per week

*Education Features:*
- `education`: Categorical education level (Bachelors, HS-grad, Masters, Doctorate, etc.)
- `education-num`: Integer encoding of education level

*Financial Features:*
- `capital-gain`: Integer, capital gains
- `capital-loss`: Integer, capital losses
- `fnlgt`: Integer, final weight (census sampling weight)

*Family Features:*
- `marital-status`: Categorical (Married-civ-spouse, Divorced, Never-married, etc.)
- `relationship`: Categorical (Wife, Husband, Not-in-family, Own-child, etc.)

**Data Preprocessing:**
1. **Cleaning:** Removed leading/trailing whitespace from all string values
2. **Feature Encoding:** 
   - One-hot encoding for 8 categorical features
   - Label binarization for target variable (0=≤$50K, 1=>$50K)
3. **Missing Values:** Handled as "?" category in categorical features
4. **No normalization** of continuous features (RandomForest is scale-invariant)

## Evaluation Data

The model is evaluated on a held-out test set of 6,513 samples (20% of total data). The test set maintains the same class distribution as the training set and was created using stratified random sampling with random_state=42 for reproducibility.

**Test Set Characteristics:**
- Same preprocessing pipeline as training data
- Uses fitted encoders from training (no data leakage)
- Representative of the full dataset distribution

## Metrics

### Overall Performance on Test Set

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision** | 0.7419 | When model predicts >$50K, it's correct 74.2% of the time |
| **Recall** | 0.6384 | Model identifies 63.8% of actual high earners |
| **F1-Score** | 0.6863 | Harmonic mean of precision and recall: 68.6% |

**Confusion Matrix Interpretation:**
- The model is more conservative (higher precision, lower recall)
- It's more likely to miss high earners (false negatives) than incorrectly predict them (false positives)
- This is common with imbalanced datasets where the positive class (>$50K) is the minority

### Performance on Data Slices

The model shows varying performance across demographic groups. Here are key findings:

**By Education Level:**
- **Best Performance:** Advanced degrees (Doctorate: F1=0.88, Masters: F1=0.84, Prof-school: F1=0.89)
- **Moderate Performance:** Bachelors (F1=0.74), Some-college (F1=0.59)
- **Poor Performance:** Elementary education levels (7th-8th: F1=0.00)
- **Insight:** Model performs significantly better for higher education levels

**By Sex:**
- Male: Precision=0.74, Recall=0.66, F1=0.70
- Female: Precision=0.72, Recall=0.52, F1=0.60
- **Concern:** ~10 percentage point F1-score gap suggests potential gender bias

**By Race:**
- Asian-Pac-Islander: F1=0.75 (highest)
- White: F1=0.69
- Black: F1=0.67
- **Concern:** Performance varies across racial groups, with some groups having fewer samples

**By Work Class:**
- Federal-gov: F1=0.79 (highest)
- State-gov: F1=0.71
- Private: F1=0.69
- Self-employed: F1=0.58-0.77 (varies by type)

**By Occupation:**
- Executive/Managerial: F1=0.77
- Professional specialty: F1=0.78
- Service workers: F1=0.32 (poorest performance)
- **Concern:** Significant performance gap between white-collar and service occupations

## Ethical Considerations

**Bias and Fairness Concerns:**

1. **Historical Bias:** The 1994 dataset reflects historical wage gaps and societal biases. The model has learned and may perpetuate:
   - Gender wage gap (lower performance for women)
   - Racial income disparities
   - Educational privilege correlations

2. **Representational Harm:** 
   - Some groups have very few samples (e.g., some countries, certain occupations)
   - Model may perform unreliably for underrepresented groups
   - Could reinforce stereotypes about income based on demographics

3. **Protected Attributes:**
   - Model uses `sex`, `race`, and `native-country` as features
   - While this increases accuracy, it raises concerns about:
     - Disparate impact
     - Potential for discriminatory use
     - Violation of fair lending/hiring laws if misused

4. **Real-World Deployment Risks:**
   - If used in hiring: Could discriminate based on protected characteristics
   - If used in lending: Could violate Equal Credit Opportunity Act
   - If used in housing: Could violate Fair Housing Act
   - **Strong recommendation:** DO NOT use for any decision affecting real people

**Privacy Considerations:**
- Dataset is public and anonymized
- No individual-level identification possible
- Aggregated statistics only

**Environmental Impact:**
- Model training: Minimal (RandomForest on small dataset)
- Carbon footprint: Negligible compared to large neural networks

## Caveats and Recommendations

### Data Limitations

1. **Temporal Shift:** Dataset is from 1994
   - Economy, demographics, and income patterns have changed significantly
   - Model predictions may not reflect current income distributions
   - Would need retraining on recent data for modern applicability

2. **Geographic Limitation:** Primarily US Census data
   - May not generalize to other countries
   - Economic conditions vary globally

3. **Class Imbalance:** 76% negative class, 24% positive class
   - Model may be biased toward predicting ≤$50K
   - Consider resampling techniques for better balance

4. **Missing Values:** Encoded as "?" category
   - May not be optimal handling strategy
   - Consider imputation methods

### Model Limitations

1. **Feature Engineering:** Uses raw features with minimal engineering
   - Could benefit from interaction terms
   - Age-education interactions might be informative

2. **Hyperparameter Tuning:** Uses default RandomForest parameters
   - No grid search or optimization performed
   - Performance could likely be improved

3. **Model Selection:** Only RandomForest tested
   - Gradient boosting (XGBoost, LightGBM) might perform better
   - No model comparison performed

4. **Interpretability:** RandomForest is a black box
   - Consider SHAP values for feature importance
   - Partial dependence plots could reveal relationships

### Recommendations for Use

**If using this model for learning:**
1. ✅ Great for understanding ML pipelines
2. ✅ Good for learning API deployment
3. ✅ Useful for studying bias in ML
4. ✅ Excellent for CI/CD practice

**If considering production use:**
1. ❌ DO NOT use for real-world decisions affecting people
2. ⚠️ If adapting the approach:
   - Collect recent, representative data
   - Conduct thorough fairness audit
   - Implement bias mitigation techniques
   - Get legal review for compliance
   - Establish human oversight
   - Monitor for disparate impact

**Bias Mitigation Strategies:**
- Remove protected attributes (sex, race) if legally required
- Apply fairness constraints during training
- Use techniques like reweighing or prejudice remover
- Implement post-processing for equalized odds
- Regular audits with tools like AI Fairness 360

### Maintenance and Update Plan

**Monitoring:**
- Track prediction distribution over time
- Monitor performance metrics by demographic slice
- Set up alerts for significant performance degradation

**Update Triggers:**
- Significant economic changes (recession, major policy changes)
- Performance drops below threshold (F1 < 0.60)
- Detected bias increase in production logs
- Availability of newer, more representative data

**Update Frequency:**
- Recommended: Annually if deployed
- Minimum: Every 5 years to account for economic shifts

**Version Control:**
- Track model versions with DVC
- Document all changes in model card
- Maintain backward compatibility in API

## Additional Information

**Contact Information:**
- GitHub: [Project Repository](https://github.com/Abdulaziz-97/Project-MLops-)
- For questions: Open an issue on GitHub

**Related Resources:**
- [UCI Dataset Documentation](https://archive.ics.uci.edu/ml/datasets/census+income)
- [Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit)
- [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)

**Citation:**
```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. 
Irvine, CA: University of California, School of Information and Computer Science.
Adult Income Dataset. https://archive.ics.uci.edu/ml/datasets/census+income
```

**Model Card Version:** 1.0  
**Last Updated:** October 2025

