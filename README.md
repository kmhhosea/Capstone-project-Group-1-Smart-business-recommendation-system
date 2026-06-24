# Smart Business Recommender System for Tanzania

Capstone project for Group 1: an educational AI/data science system that recommends promising business/product categories for Tanzanian regions using cleaned prototype data, machine learning, visualization, Gradio, and an OpenAI-powered explanation layer.

GitHub repository:
https://github.com/kmhhosea/Capstone-project-Group-1-Smart-business-recommendation-system

## Project Goal

The project helps explain how data science can support small business decision-making. A user selects a Tanzanian region, month, rainfall level, temperature level, customer group, shop type, and age. The trained model recommends product categories and top products, while the LLM gives practical age-aware business advice.

## Why This Project Matters

- It shows the full data science workflow from messy data to a deployed prototype.
- It uses beginner-friendly tools learned in data science and AI coursework.
- It demonstrates why data cleaning, EDA, model evaluation, and responsible interpretation matter.
- It uses Gradio as a simple Python library for quickly building a web interface.
- It includes LLM interpretation to make model outputs easier for non-technical users.

## Folder Structure

```text
CAPSTONE-PROJECT/
  data/
    raw/
      manual_assembled_unclean_business_data.xlsx
    processed/
      cleaned_business_data_stage6.csv
    final/
      project_ready_cleaned_for_ml.csv
      project_ready_personalized_tanzania_data_cleaned.xlsx
      model_comparison_results.csv
  models/
    smart_business_recommender_model.joblib
  notebooks/
    01_data_preparation_stage_1_to_6.ipynb
    02_ml_training_evaluation_gradio_stage_7_to_13.ipynb
    
  .gitignore
  README.md
```

## Notebooks

1. `01_data_preparation_stage_1_to_6.ipynb`

   Covers problem definition, data collection story, data understanding, cleaning, EDA, and preprocessing. It starts unclean manually assembled data and produces clean ML-ready data.

2. `02_ml_training_evaluation_gradio_stage_7_to_13.ipynb`

   Covers feature selection, train/test split, several ML algorithms, model evaluation, tuning, model saving, and final Gradio deployment. The final interface uses only the project dataset and does not ask users to upload spreadsheets.

## Final Interface

The final Gradio app accepts:

- Tanzania region
- User age
- Month
- Rainfall level: Low, Medium, High
- Temperature level: Low, Medium, High
- Customer group
- Shop type

It outputs:

- Recommended product categories
- Top 10 products in the best category
- Pie chart of recommendation scores
- LLM business advice based on the user's age



## How To Run

1. Open `notebooks/01_data_preparation_stage_1_to_6.ipynb` and run all cells.
2. Open `notebooks/02_ml_training_evaluation_gradio_stage_7_to_13.ipynb` and run all cells.
3. The final cell launches the Gradio interface.


