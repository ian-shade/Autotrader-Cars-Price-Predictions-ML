# Autotrader Cars Price Predictions - Machine Learning

A comprehensive machine learning project for predicting used car prices from Autotrader UK data using regression models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ian-shade/Autotrader-Cars-Price-Predictions-ML/blob/main/ml_project.ipynb)

## Project Overview

This project analyzes 400,000+ car listings from Autotrader to build predictive models for estimating vehicle prices. Through extensive feature engineering, data cleaning, and model optimization, the final Decision Tree model achieves an R² score of 0.89 on the test set.

## Dataset

- **Size**: 402,005 car listings
- **Source**: Autotrader UK (adverts.csv)
- **Features**: 12 original features including mileage, make, model, color, body type, fuel type, condition, and year

### Key Features

**Numeric Features**:
- `mileage`: Vehicle mileage in miles
- `age`: Vehicle age (engineered from year of registration)
- `brand_prestige`: Custom classification (1-5 scale) based on manufacturer luxury tier

**Categorical Features**:
- `standard_make`: Car manufacturer (top 20 brands retained)
- `standard_colour`: Vehicle color (top 10 colors retained)
- `body_type`: Vehicle body style (SUV, Saloon, Hatchback, etc.)
- `fuel_type`: Fuel type (Petrol, Diesel, Hybrid, Electric)
- `vehicle_condition`: NEW or USED

## Feature Engineering

### Brand Prestige Classification

Created a custom 5-tier prestige system based on manufacturer positioning:
- **Class 5**: Exotic/Hypercar (Ferrari, Lamborghini, McLaren) - avg £150k
- **Class 4**: Ultra-Luxury (Bentley, Rolls-Royce) - avg £95k
- **Class 3**: Luxury (Jaguar, Land Rover, Porsche) - avg £40k
- **Class 2**: Premium (Audi, BMW, Mercedes-Benz) - avg £20k
- **Class 1**: Mass Market (Ford, Toyota, Vauxhall) - avg £10k

### Data Cleaning

- Removed vintage cars (>33 years old) to avoid appreciation dynamics
- Filtered extreme mileage outliers (>300k miles) to exclude taxis/errors
- Capped prices at £350k to remove super-luxury outliers
- Dropped near-zero variance features (`crossover_car_and_van`)
- Consolidated low-frequency categories into "Other" groups
- Applied log transformation to target variable for better model performance

## Model Performance

### Models Evaluated

Three regression models were optimized using GridSearchCV:

| Model | Best Parameters | CV R² | Test R² |
|-------|----------------|-------|---------|
| **Decision Tree** | depth=20, split=50, leaf=10 | 0.892 | 0.894 |
| Ridge Regression | alpha=10 | 0.852 | 0.852 |
| k-NN | k=7, manhattan, distance | 0.799 | 0.884 |

### Final Model: Decision Tree Regressor

**Performance Metrics**:
- **R² Score**: 0.89 (log-transformed), 0.81 (real prices)
- **Mean Absolute Error (MAE)**: £3,352
- **Root Mean Squared Error (RMSE)**: £7,913
- **Train-Test Gap**: 0.018 (minimal overfitting)

**Top Feature Importances**:
1. Age (0.45)
2. Mileage (0.32)
3. Brand Prestige (0.12)
4. Vehicle Condition (0.03)
5. Body Type & Fuel Type (combined 0.08)

## Key Insights

- **Age is the strongest predictor**: Depreciation heavily impacts pricing
- **Mileage matters significantly**: High mileage reduces value substantially
- **Brand prestige creates distinct pricing tiers**: Luxury brands maintain higher values
- **Model struggles with ultra-luxury segment**: Underestimates £200k+ vehicles due to limited training data (<1%)
- **Log transformation essential**: Original price distribution was heavily right-skewed

## Project Structure

```
ml_project.ipynb
├── 1. Data Understanding & Processing
│   ├── 1.1 Initial Data Inspection
│   ├── 1.2 Data Quality Fixes
│   ├── 1.3 Target Feature Analysis
│   ├── 1.4 Numeric Features Analysis
│   ├── 1.5 Brand Prestige Engineering
│   └── 1.6 Visual Analysis
├── 2. Data Processing for ML
│   ├── 2.1 One-Hot Encoding
│   ├── 2.2 Train/Test Split (80/20)
│   └── 2.3 Feature Scaling
├── 3. Model Building
│   ├── 3.1 Cross Validation
│   └── 3.2 Grid Search Optimization
│       ├── Ridge Regression
│       ├── Decision Tree
│       └── k-NN
└── 4. Final Model Evaluation
    ├── 4.1 Model Training
    ├── 4.2 Performance Metrics
    ├── 4.3 Feature Importance
    └── 4.4 Error Analysis
```

## Requirements

```python
pandas
numpy
seaborn
matplotlib
scikit-learn
google.colab (for Google Drive integration)
```

## Usage

### Google Colab (Recommended)

1. Click the "Open in Colab" badge at the top
2. Mount your Google Drive
3. Upload `adverts.csv` to `/content/drive/MyDrive/Colab/`
4. Run all cells sequentially

### Local Environment

```bash
# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn

# Update file path in notebook
file_path = 'path/to/adverts.csv'

# Run the notebook
jupyter notebook ml_project.ipynb
```

## Limitations & Future Work

**Current Limitations**:
- Underperforms on ultra-luxury segment (£200k+) due to data scarcity
- Geographic pricing variations not captured
- Seasonal market trends not included
- Missing features: accident history, service records, trim levels

**Potential Improvements**:
- Collect more high-end vehicle data or build separate luxury model
- Incorporate regional pricing factors
- Add temporal features (listing date, seasonality)
- Ensemble methods (Random Forest, XGBoost) for potential performance gains
- Feature interaction terms (e.g., age × brand prestige)

## License

This project is open source and available under the MIT License.

## Author

Created for machine learning portfolio demonstration