# IMU-Based Fall Detection System Using Machine Learning

## Project Overview

This repository implements a high-precision machine learning system for real-time fall detection using wearable Inertial Measurement Unit (IMU) sensors. The system leverages advanced signal processing techniques and ensemble learning algorithms to achieve state-of-the-art detection accuracy while maintaining robustness across different subjects and scenarios.

## Technical Architecture

### Data Acquisition

- **Sensor platform**: Multi-modal IMU sensors (accelerometer, gyroscope, magnetometer)
- **Mounting positions**: Head, chest, waist, right wrist, right thigh, right ankle
- **Sampling rate**: 100Hz (configurable)
- **Dataset dimensions**: 3,324 samples (1,842 falls, 1,482 non-falls)
- **Data structure**: Each sample contains synchronized multi-dimensional time series from all available sensors

### Signal Processing & Feature Engineering

The system implements a comprehensive feature extraction pipeline that converts raw sensor data into a 58-dimensional feature space optimized for fall detection:

```
feature_dict = {
    'sample_id': sample_id,
    'participant_id': sample['participant_id'],
    'is_fall': is_fall
}
```

#### Time-Domain Features
- Statistical moments (mean, std, min, max, range, percentiles)
- Zero-crossing rates
- Acceleration magnitude calculations
- Signal Magnitude Area (SMA)
- Jerk calculations (time derivative of acceleration)
- Axis correlations

#### Frequency-Domain Features
- Fast Fourier Transform (FFT) components
- Dominant frequency identification
- Frequency-domain energy

#### Physics-Based Features
- Angular velocity peak-to-peak measurements
- Orientation features (roll, pitch, yaw)
- Vector magnitude calculations

### Machine Learning Implementation

The core classification pipeline employs:

- **Algorithm**: Random Forest Classifier (100 estimators)
- **Validation strategy**: Participant-independent cross-validation (5-fold)
- **Feature normalization**: Standardization (zero mean, unit variance)
- **Data partitioning**: 14 participants for training, 3 participants for testing

### Performance Optimization

The feature selection process identified the top predictive features:

![Top 20 Most Important Features](Top 20 Most Important Features.png)

Key feature importance findings:
1. `waist_acc_mag_max`: 15.3% (maximum acceleration magnitude)
2. `waist_acc_mag_range`: 13.1% (range of acceleration magnitude)
3. `waist_acc_mag_min`: 10.3% (minimum acceleration magnitude)
4. `waist_gyr_mag_p2p`: 9.2% (gyroscope magnitude peak-to-peak)
5. `waist_gyr_mag_max`: 7.0% (maximum gyroscope magnitude)

## Experimental Results

### Binary Classification Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 99.64% |
| Precision (Fall) | 100.0% |
| Recall (Fall) | 99.0% |
| F1-Score (Fall) | 99.5% |
| Precision (Non-Fall) | 99.2% |
| Recall (Non-Fall) | 100.0% |
| F1-Score (Non-Fall) | 99.6% |

### Cross-Validation Results

| Fold | Accuracy | Precision (Fall) | Recall (Fall) |
|------|----------|------------------|---------------|
| 1 | 99.74% | 100.0% | 99.8% |
| 2 | 98.29% | 98.1% | 98.8% |
| 3 | 97.73% | 97.1% | 98.8% |
| 4 | 98.96% | 99.0% | 99.1% |
| 5 | 99.16% | 99.7% | 98.8% |
| **Mean** | **98.78%** | **98.78%** | **99.06%** |

### Confusion Matrix (Test Set)

|               | Predicted Non-Fall | Predicted Fall |
|---------------|-------------------|----------------|
| Actual Non-Fall | 259 (TP) | 0 (FP) |
| Actual Fall   | 2 (FN) | 300 (TN) |

## Technical Innovations

1. **Participant-independent validation**: Implemented a rigorous participant-based split to ensure that data from the same participant doesn't appear in both training and test sets, resulting in a more realistic assessment of model generalization.

2. **Optimized sensor selection**: Demonstrated that a single waist-mounted sensor can achieve comparable performance to multi-sensor setups, reducing system complexity while maintaining detection accuracy.

3. **Feature importance analysis**: Quantified the relative importance of different feature types, showing that acceleration magnitude features contribute most significantly to classification performance.

4. **Cross-validated performance assessment**: Established robustness through 5-fold cross-validation with different participant subsets, validating generalization capability to unseen subjects.

## Implementation Details

### Data Preprocessing

```python
def preprocess_fall_detection_data_binary(features_df, labels_series, normalization='standard', 
                                  test_size=0.2, random_state=42):
    # Create binary labels (1 for fall, 0 for non-fall)
    binary_labels = labels_series.apply(lambda x: 1 if str(x).startswith('9') else 0)
    
    # Get metadata columns (non-feature columns to exclude from normalization)
    metadata_cols = ['sample_id', 'participant_id', 'is_fall']
    feature_cols = [col for col in features.columns if col not in metadata_cols]
    
    # Handle missing values and apply normalization
    if normalization == 'standard':
        scaler = StandardScaler()
        features[feature_cols] = scaler.fit_transform(features[feature_cols])
    
    # Participant-based train/test split
    participants = features_df['participant_id'].unique()
    n_test = int(len(participants) * test_size)
    test_participants = participants[:n_test]
    train_participants = participants[n_test:]
    
    # Create masks for train and test sets
    train_mask = features_df['participant_id'].isin(train_participants)
    test_mask = features_df['participant_id'].isin(test_participants)
    
    # Split data
    X_train = features_df.loc[train_mask, feature_cols]
    X_test = features_df.loc[test_mask, feature_cols]
    y_train = features_df.loc[train_mask, 'is_fall']
    y_test = features_df.loc[test_mask, 'is_fall']
    
    return X_train, X_test, y_train, y_test, scaler
```

### Model Training

```python
# Train a Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf_model.fit(X_train, y_train)
```

### Participant-Independent Cross-Validation

```python
def participant_cross_validation(features_df, model, n_splits=5, random_state=42):
    # Get unique participants
    participants = features_df['participant_id'].unique()
    
    # Create KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_scores = []
    
    # Perform cross-validation
    for train_idx, test_idx in kf.split(participants):
        # Get train and test participants
        train_participants = participants[train_idx]
        test_participants = participants[test_idx]
        
        # Create masks for train and test sets
        train_mask = features_df['participant_id'].isin(train_participants)
        test_mask = features_df['participant_id'].isin(test_participants)
        
        # Split data
        X_train = features_df.loc[train_mask, feature_cols]
        X_test = features_df.loc[test_mask, feature_cols]
        y_train = features_df.loc[train_mask, 'is_fall']
        y_test = features_df.loc[test_mask, 'is_fall']
        
        # Train and evaluate model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores.append(accuracy)
    
    return cv_scores
```

## Conclusion

This implementation demonstrates a highly accurate fall detection system using a single waist-mounted IMU sensor. The system achieves 99.64% accuracy on the test set and 98.78% mean accuracy across 5-fold cross-validation. The participant-independent validation approach ensures that the model generalizes well to new users, making it suitable for real-world deployment.

The high precision (100.0%) and recall (99.0%) for fall detection indicate that the system has both an extremely low false alarm rate and a high detection rate for actual falls, addressing the two critical requirements for practical fall detection systems.

## Technologies Used

- **Languages**: Python 3.8+
- **Libraries**: 
  - scikit-learn 1.0+
  - NumPy 1.20+
  - pandas 1.3+
  - SciPy 1.7+
  - Matplotlib 3.4+
  - seaborn 0.11+
- **Development Tools**:
  - Jupyter Notebook
  - Git/GitHub
