'''
## Stage 13 — Model Deployment with Gradio

The final interface uses the cleaned project dataset and trained model. 
The user enters business context such as region, customer group, shop type, 
rainfall level, temperature level, and age. The model recommends business/product 
categories, while the LLM uses the user's age to explain suitable business 
opportunities in simple language.


'''

import joblib
import pandas as pd
import matplotlib.pyplot as plt
USE_LLM = True  

# # Load your data (adjust path as needed)
# df = pd.read_csv("data/processed/cleaned_business_data_stage6.csv")

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "data" / "processed" / "cleaned_business_data_stage6.csv"
)

best_model = joblib.load(
    ROOT / "models" / "smart_business_recommender_model.joblib"
)


from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it",
    device_map="auto"
)



def value_from_level(column, level):
    if level == "Low":
        return float(df[column].quantile(0.25))
    if level == "Medium":
        return float(df[column].quantile(0.50))
    return float(df[column].quantile(0.75))

def high_probability(model, row):
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    if "High" in classes:
        return float(probabilities[classes.index("High")])
    return 0.0

def age_group_from_age(age):
    age = int(age)
    if age < 25:
        return "young entrepreneur/student age group"
    if age < 40:
        return "early-career adult age group"
    if age < 60:
        return "experienced adult age group"
    return "senior/experienced community member age group"




def local_business_explanation(best_category, top_products, region, rainfall_level, temperature_level, age):
    products_text = ", ".join(top_products[:5])
    age_group = age_group_from_age(age)
    return (
        f"Recommended business: start or stock a {best_category} business in {region}. "
        f"For a {age_group}, this opportunity can be approached according to available energy, capital, and experience. "
        f"The model ranked this category highest for {rainfall_level.lower()} rainfall and "
        f"{temperature_level.lower()} temperature. Good products to begin with are: {products_text}."
    )

def llm_business_explanation(
    best_category,
    top_products,
    region,
    rainfall_level,
    temperature_level,
    customer_group,
    shop_type,
    age
):
    fallback = local_business_explanation(
        best_category,
        top_products,
        region,
        rainfall_level,
        temperature_level,
        age
    )

    try:
        age_group = age_group_from_age(age)

        prompt = f"""
Explain this smart business recommendation in simple language for a person in Tanzania.

The person is {age} years old and belongs to the {age_group}.

Region: {region}
Customer group: {customer_group}
Shop type: {shop_type}
Rainfall level: {rainfall_level}
Temperature level: {temperature_level}

Recommended business category:
{best_category}

Top products:
{", ".join(top_products[:10])}

Explain:
1. Why this business may fit.
2. How age affects risk and management.
3. First practical steps.
4. Reminder that profit is not guaranteed.

Keep it short and suitable for a school capstone presentation.
"""

        response = generator(
            prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7
        )

        generated = response[0]["generated_text"]

        return generated[len(prompt):].strip()

    except Exception as error:
        return fallback + f"\n\nLLM fallback used because: {error}"
    






def recommend_business(region, month, rainfall_level, temperature_level, customer_group, shop_type, age):
    training_data = df.copy()
    current_model = tuned_model if "tuned_model" in globals() else best_model

    rainfall_value = value_from_level("rainfall_mm", rainfall_level)
    temperature_value = value_from_level("temperature_c", temperature_level)

    category_rows = []
    for category in sorted(training_data["product_category"].unique()):
        category_data = training_data[training_data["product_category"] == category]
        common_product = category_data["product_name"].mode().iloc[0]
        typical_price = category_data["unit_price_tzs"].median()
        test_row = pd.DataFrame([{
            "month": month,
            "region": region,
            "rainfall_mm": rainfall_value,
            "temperature_c": temperature_value,
            "customer_group": customer_group,
            "shop_type": shop_type,
            "product_category": category,
            "product_name": common_product,
            "unit_price_tzs": typical_price,
        }])
        category_rows.append({
            "product_category": category,
            "example_product": common_product,
            "high_demand_probability": high_probability(current_model, test_row),
        })

    category_result = pd.DataFrame(category_rows).sort_values("high_demand_probability", ascending=False)
    best_category = category_result.iloc[0]["product_category"]

    product_result = (
        training_data[training_data["product_category"] == best_category]
        .groupby("product_name", as_index=False)
        .agg(avg_units_sold=("units_sold", "mean"), median_price_tzs=("unit_price_tzs", "median"))
        .sort_values("avg_units_sold", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    pie_data = category_result.head(5)
    ax.pie(
        pie_data["high_demand_probability"],
        labels=pie_data["product_category"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Top Category Recommendation Scores")

    if USE_LLM:
        explanation = llm_business_explanation(
        best_category,
        product_result["product_name"].tolist(),
        region,
        rainfall_level,
        temperature_level,
        customer_group,
        shop_type,
        age,
    )
    else:
        explanation = ""

    return category_result, product_result, fig, explanation











import gradio as gr

region_choices = sorted(df["region"].unique().tolist())
customer_choices = sorted(df["customer_group"].unique().tolist())
shop_choices = sorted(df["shop_type"].unique().tolist())

with gr.Blocks(title="Smart Business Recommender for Tanzania") as demo:
    gr.HTML(
        '''
        <style>
        body { background: #fff8f0; }
        .gradio-container { background-color: #fff8f0; }
        .gr-button { background-color: orange !important; color: white !important; border-color: darkorange !important; }
        .gr-button:hover { background-color: darkorange !important; }
        .gr-textbox, .gr-dropdown, .gr-dataframe, .gradio-plot { border-radius: 10px; }
        .gr-markdown { color: #302b27; }
        </style>
        '''
    )

    gr.Markdown("# Smart Business Recommender for Tanzania")
    gr.Markdown(
        "Choose a Tanzanian business context. The system uses our cleaned project dataset and trained ML model, "
        "then the LLM explains the recommendation using the user's age."
    )

    with gr.Row():
        region_input = gr.Dropdown(region_choices, label="Tanzania Region", value=region_choices[0])
        age_input = gr.Slider(16, 75, value=25, step=1, label="Your Age")

    with gr.Row():
        month_input = gr.Dropdown([1, 4, 7, 10], label="Month", value=1)
        rainfall_input = gr.Dropdown(["Low", "Medium", "High"], label="Rainfall Level", value="Medium")
        temperature_input = gr.Dropdown(["Low", "Medium", "High"], label="Temperature Level", value="Medium")

    with gr.Row():
        customer_input = gr.Dropdown(customer_choices, label="Customer Group", value=customer_choices[0])
        shop_input = gr.Dropdown(shop_choices, label="Shop Type", value=shop_choices[0])

    recommend_button = gr.Button("Get Recommendation")

    category_output = gr.Dataframe(label="Recommended Product Categories")
    product_output = gr.Dataframe(label="Top 10 Products in Best Category")
    pie_output = gr.Plot(label="Recommendation Pie Chart")
    explanation_output = gr.Textbox(label="LLM Business Advice Based on Age", lines=10)

    recommend_button.click(
        recommend_business,
        inputs=[region_input, month_input, rainfall_input, temperature_input, customer_input, shop_input, age_input],
        outputs=[category_output, product_output, pie_output, explanation_output],
    )

demo.launch()