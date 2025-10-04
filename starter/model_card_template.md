# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Model Type:** Random Forest Classifier  
**Model Version:** 1.0  
**Date:** October 2025  
**Author:** MLOps Project Team  

The model is a Random Forest ensemble classifier implemented using scikit-learn with 100 estimators and a fixed random state (42) for reproducibility. It uses one-hot encoding for categorical features and label binarization for the target variable.

## Intended Use

**Primary Use Cases:**
- Educational demonstration of MLOps practices
- Salary prediction research and analysis
- Understanding demographic factors affecting income

**Intended Users:**
- Data scientists and ML engineers learning MLOps
- Researchers studying income inequality
- Students learning machine learning concepts

**Out-of-Scope Use Cases:**
- Production hiring or compensation decisions
- Individual financial assessments
- Any decision-making that could impact real individuals' lives

## Training Data

**Dataset:** UCI Adult Census Income Dataset  
**Source:** https://archive.ics.uci.edu/ml/datasets/census+income  
**Size:** 32,561 samples with 15 features  
**Target Distribution:**
- <=50K: 76.07% (24,720 samples)
- >50K: 23.93% (7,841 samples)

**Features Used:**
- **Demographic:** age, race, sex, native-country
- **Work-related:** workclass, occupation, hours-per-week
- **Education:** education, education-num
- **Financial:** capital-gain, capital-loss, fnlgt
- **Family:** marital-status, relationship

**Data Preprocessing:**
- Removed leading/trailing spaces from all string values
- One-hot encoded categorical features (8 features)
- Label binarized target variable
- Train-test split: 80% training (26,048 samples), 20% testing (6,513 samples)

## Evaluation Data

**Test Set:** 6,513 samples (20% of total data)  
**Split Method:** Stratified random split with random_state=42  
**Distribution:** Maintains similar class distribution to training data

## Metrics

**Overall Performance on Test Set:**
- **Precision:** 0.7419 (74.19%)
- **Recall:** 0.6384 (63.84%)
- **F1-Score:** 0.6863 (68.63%)

**Interpretation:**
- **Precision:** When the model predicts someone makes >$50K, it's correct 74% of the time
- **Recall:** The model correctly identifies 64% of people who actually make >$50K
- **F1-Score:** Balanced measure showing overall model performance of 69%

**Performance by Demographic Groups:**

*By Education Level:*
- Prof-school (F1: 0.8852), Doctorate (F1: 0.8793), Masters (F1: 0.8409) - Best performance
- 7th-8th grade (F1: 0.0000) - Poorest performance due to class imbalance

*By Sex:*
- Male (F1: 0.6997) - Slightly better performance  
- Female (F1: 0.6015) - Lower performance, likely due to historical income disparities

*By Race:*
- Asian-Pac-Islander (F1: 0.7458) - Highest performance
- White (F1: 0.6850) - Close to overall average
- Performance varies across racial groups, reflecting dataset biases

## Ethical Considerations

**Bias Analysis:**
- The model shows differential performance across demographic groups
- Historical biases in the dataset (1994 Census data) are reflected in predictions
- Gender bias evident: lower recall for women may perpetuate wage gap assumptions
- Racial disparities in performance could lead to unfair predictions

**Fairness Concerns:**
- The model may reinforce existing societal biases about income based on demographics
- Not suitable for any decision-making affecting real individuals
- Performance gaps across groups indicate potential discriminatory outcomes

**Privacy:**
- Uses aggregated census data, no individual personal information
- Predictions should not be used to make inferences about specific individuals

**Transparency:**
- Model weights and feature importance can be inspected
- Slice analysis provided to understand group-level performance
- All code and methodology documented for reproducibility

## Caveats and Recommendations

**Data Limitations:**
- Training data from 1994 - may not reflect current economic conditions
- Limited geographic scope (primarily US-based)
- Missing values encoded as "?" may introduce noise
- Highly imbalanced classes (76% vs 24%) affect model performance

**Model Limitations:**
- Random Forest may not capture complex feature interactions optimally
- Performance varies significantly across demographic subgroups
- Lower recall (63.84%) means many high earners are misclassified as low earners

**Recommendations for Use:**
- Use only for educational and research purposes
- Always analyze performance across demographic groups when deploying
- Consider bias mitigation techniques for any adapted production use
- Regular retraining with updated data would improve relevance
- Implement fairness constraints if adapting for any real-world application

**Recommendations for Improvement:**
- Collect more recent training data
- Apply bias mitigation techniques (e.g., fairness constraints, resampling)
- Try alternative algorithms (e.g., XGBoost, neural networks)
- Feature engineering to better capture economic factors
- Ensemble methods combining multiple model types

**Monitoring Recommendations:**
- Track prediction performance across demographic groups over time
- Monitor for dataset drift if deploying with newer data
- Regular bias audits to ensure fair treatment across populations
