# Singapore HDB Resale Price Prediction 🏘️

A machine learning project that predicts Singapore HDB (Housing & Development Board) resale prices using advanced feature engineering and multi-task learning architecture. The model leverages geospatial data and implements a novel multi-task approach to improve prediction accuracy over traditional single-task models.

## 🎯 Project Overview

This project addresses the challenge of accurately predicting HDB resale prices in Singapore by:
- Integrating external geospatial features through web scraping
- Implementing a multi-task learning architecture that jointly predicts price buckets and exact prices
- Leveraging shared representations to improve model robustness and generalization

## ✨ Key Features

### Advanced Feature Engineering
- **Geospatial Data Integration**: Web-scraped location-based features using OneMap.sg API
- **Distance Calculations**: Computed proximity metrics using GeoPy library (e.g., distance to MRT stations, schools, amenities)
- **Contextual Features**: Incorporated neighborhood characteristics and environmental factors
- **Feature Optimization**: Systematic feature selection and engineering to enhance predictive power

### Multi-Task Learning Architecture
- **Dual Output Heads**: 
  - Task 1: Classification of resale prices into buckets (price ranges)
  - Task 2: Regression for exact price prediction
- **Shared Representations**: Common feature layers that learn generalizable patterns across both tasks
- **Performance Improvement**: Outperformed single-task baseline models through knowledge transfer between related tasks

## 🛠️ Technologies Used

- **Machine Learning**: Scikit-learn, [TensorFlow/PyTorch - specify which you used]
- **Data Processing**: Pandas, NumPy
- **Geospatial Analysis**: GeoPy
- **Web Scraping**: Requests, BeautifulSoup (for OneMap.sg API)
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebook / Google Colab

## 📊 Dataset

- **Source**: [Singapore Open Data Portal / data.gov.sg - specify your source]
- **Features**: Flat type, floor area, storey range, lease commence date, town, location coordinates, and engineered geospatial features
- **Target Variable**: HDB resale price (in SGD)

## 🏗️ Methodology

### 1. Data Collection & Preprocessing
- Collected historical HDB resale transaction data
- Web-scraped complementary geospatial data from OneMap.sg API
- Handled missing values and outliers
- Normalized numerical features and encoded categorical variables

### 2. Feature Engineering
- **Location-Based Features**:
  - Distance to nearest MRT station
  - Proximity to schools, malls, parks
  - Town and planning area information
- **Property Features**:
  - Remaining lease duration
  - Floor level categorization
  - Flat type and size specifications
- **Temporal Features**:
  - Transaction year and month
  - Age of the property

### 3. Multi-Task Learning Model
```
Input Features
      ↓
Shared Layers (Feature Extraction)
      ↓
   ┌──────┴──────┐
   ↓             ↓
Task 1:       Task 2:
Price Bucket  Exact Price
(Classifier)  (Regressor)
```

**Architecture Benefits**:
- **Knowledge Transfer**: Classification task helps regression by learning price ranges
- **Regularization Effect**: Multi-task learning reduces overfitting
- **Improved Robustness**: Shared layers learn more generalizable features

### 4. Model Evaluation
- Compared against single-task baselines (Linear Regression, Random Forest, XGBoost)
- Evaluation metrics: RMSE, MAE, R² for regression; Accuracy, F1-score for classification
- Cross-validation for robust performance estimation

## 📈 Results

- **Multi-task model outperformed single-task baselines** in both price prediction accuracy and price bucket classification
- **Feature engineering significantly improved model accuracy** by incorporating location-specific context
- Key predictive features: [Add top features - e.g., floor area, remaining lease, distance to MRT, town]

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn geopy requests
# Add TensorFlow or PyTorch if using deep learning
```

### Running the Notebook
1. Clone the repository:
```bash
git clone https://github.com/jaredteoh/Singapore-HDB-Resale-Price-Prediction.git
cd Singapore-HDB-Resale-Price-Prediction
```

2. Open the Jupyter notebook:
```bash
jupyter notebook HDB_Resale_Price_Prediction.ipynb
```

Or run directly in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaredteoh/Singapore-HDB-Resale-Price-Prediction/blob/main/HDB_Resale_Price_Prediction.ipynb)

## 📁 Project Structure

```
Singapore-HDB-Resale-Price-Prediction/
│
├── HDB_Resale_Price_Prediction.ipynb    # Main notebook with analysis
├── data/                                # Dataset (if included)
├── models/                              # Saved models (if applicable)
└── README.md                            # Project documentation
```

## 🔍 Key Insights

- **Location matters**: Proximity to amenities significantly impacts resale prices
- **Multi-task learning advantage**: Joint training improved generalization and reduced prediction errors
- **Feature engineering impact**: Geospatial features provided substantial predictive power beyond basic property attributes

## 🚧 Future Enhancements

- [ ] Incorporate additional data sources (economic indicators, population density)
- [ ] Experiment with ensemble methods combining multiple multi-task models
- [ ] Deploy as a web application for real-time price predictions
- [ ] Add interpretability features (SHAP values, feature importance visualization)
- [ ] Implement time-series forecasting for future price trends

## 📚 References

- OneMap.sg API: Singapore's national map platform
- GeoPy Documentation: [https://geopy.readthedocs.io/](https://geopy.readthedocs.io/)
- Multi-Task Learning: Caruana, R. (1997). "Multitask Learning"

## 👤 Author

**Jared Teoh Jie Rui**
- LinkedIn: [linkedin.com/in/jaredteoh0725](https://www.linkedin.com/in/jaredteoh0725/)
- GitHub: [github.com/jaredteoh](https://github.com/jaredteoh)
- Email: teohjared@gmail.com

## 📄 License

This project is open source and available under the MIT License.

---

*Predicting Singapore's housing market with machine learning* 🏠📊
