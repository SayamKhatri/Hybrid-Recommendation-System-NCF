# Hybrid Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)

A deep learning-based recommendation system combining **Neural Collaborative Filtering (NCF)** and **content-based filtering** to deliver personalized movie recommendations. Trained on the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m/) with ~10 million user-movie interactions, achieving competitive metrics like **HR@10: 0.8788** and **NDCG@10: 0.4466**.

# Features
- **Neural Collaborative Filtering (NCF)**: Deep learning model for user-movie interaction prediction.
- **Content-Based Filtering**: Leverages movie metadata (e.g., genres) to address cold-start problem.
- **Hybrid Approach**: Combines NCF and content-based scores for robust recommendations.
- **Scalable Pipeline**: Processes ~10M+ records using TensorFlow, Keras, Pandas, and NumPy.

# Results
- **HR@10**: 0.8788 (ranks true positives in top-10 for ~88% of users)
- **NDCG@10**: 0.4466 (high ranking quality)
- **Train Acc**: Mean ~0.90 
- **Val Acc**: Mean ~0.88 (non-interactions)

# Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Hybrid-Recommendation-System.git
   cd Hybrid-Recommendation-System
   ```


2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
  
3. Download the MovieLens 10M dataset:
   ```bash
    Place files in data/ 
    ```
  
  
 # Usage
  
  #### Preprocess Data:
  ```
  python src/ncf/data_preprocessing.py
  ```
  
  #### Train NCF Model:
  ```
  python src/ncf/train.py
  ```
  
  #### Generate Recommendations:
  ```
  python src/ncf/inference.py --user_id 100
  ```
  
  #### Explore Notebooks:
  ```
  notebooks/data_exploration.ipynb: Visualize dataset.
  notebooks/model_evaluation.ipynb: Experiment with models.
  ```
  
  
# Directory Structure
  ```
  Hybrid-Recommendation-System/
  ├── data/                   # Dataset files
  ├── src/                    # Source code
  │   ├── ncf/                # NCF scripts
  │   ├── content_based/      # Content-based scripts
  │   └── hybrid/             # Hybrid model scripts
  ├── notebooks/              # Jupyter notebooks
  ├── models/                 # Trained models
  ├── requirements.txt        # Dependencies
  ├── README.md               # Project overview
  ├── .gitignore              # Ignored files
  └── LICENSE.txt             # License
  ```
# Technologies
  ```
  Python: 3.8+
  TensorFlow/Keras: 2.10+ (NCF model)
  Pandas/NumPy: Data processing
  Scikit-learn: Metrics and content-based features
  ```

# License

  This project is licensed under the MIT License (see LICENSE.txt).

