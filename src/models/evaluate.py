"""
Evaluation & Visualization Module.
Generates evaluation plots: Residuals, Predicted vs Actual, Feature Importances.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import yaml

from src.utils.logger import get_logger

logger = get_logger("evaluate")

def evaluate_and_plot(params_path: str = "params.yaml"):
    with open(params_path, 'r') as f:
        config = yaml.safe_load(f)
        
    train_cfg = config['train']
    eval_cfg = config['evaluate']
    
    input_path = train_cfg['input_path']
    model_path = eval_cfg['model_path']
    target_col = train_cfg['target_column']
    
    logger.info(f"Loading dataset for evaluation from {input_path}")
    df = pd.read_csv(input_path) if input_path.endswith('.csv') else pd.read_parquet(input_path)
    
    logger.info(f"Loading trained model from {model_path}")
    model_artifacts = joblib.load(model_path)
    model = model_artifacts['model']
    feature_names = model_artifacts['feature_names']
    model_name = model_artifacts['model_name']
    
    X = df[feature_names]
    y = df[target_col].values
    
    preds = model.predict(X)
    residuals = y - preds
    
    output_dir = Path("scorer_results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Residuals distribution plot
    plt.figure(figsize=(10, 5), dpi=150)
    sns.histplot(residuals, kde=True, color="#064A56", bins=50)
    plt.axvline(0, color="red", linestyle="--", alpha=0.7)
    plt.title(f"Residual Distribution ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Residual ($Actual - Predicted$)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    res_path = output_dir / "residuals_distribution.png"
    plt.savefig(res_path)
    plt.close()
    
    # 2. Predicted vs Actual scatter plot
    plt.figure(figsize=(8, 8), dpi=150)
    plt.scatter(y, preds, alpha=0.3, color="#064A56", s=10)
    plt.plot([0, max(y)], [0, max(y)], 'r--', label="Perfect Prediction")
    plt.title(f"Predicted vs Actual Rates ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Posted Rate ($)")
    plt.ylabel("Predicted Rate ($)")
    plt.legend()
    plt.tight_layout()
    scatter_path = output_dir / "predicted_vs_actual.png"
    plt.savefig(scatter_path)
    plt.close()
    
    # 3. Feature Importance Plot (if model supports it)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)
        
        plt.figure(figsize=(10, 6), dpi=150)
        sns.barplot(x="importance", y="feature", data=imp_df, palette="crest")
        plt.title(f"Top 15 Feature Importances ({model_name})", fontsize=14, fontweight="bold")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        imp_path = output_dir / "feature_importance.png"
        plt.savefig(imp_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {imp_path}")
        
    logger.info(f"Saved evaluation plots to {output_dir}")

if __name__ == "__main__":
    evaluate_and_plot()
