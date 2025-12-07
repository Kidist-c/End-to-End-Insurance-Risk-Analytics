# import libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# EDA Class
class InsuranceEDA:
    def __init__(self, df: pd.DataFrame):
        """Initialize the InsuranceEDA class with a processed DataFrame."""
        self.df = df
        print("=== EDA Instance Created ===\n")

    # 1. DATA SUMMARIZATION
    def data_summary(self):
        """Generate a summary of the DataFrame including data types, shape, missing values, and basic statistics."""
        print("=== DATA SUMMARY ===\n")
        print(f"Data Shape: {self.df.shape}\n")
        print(f"Data Types:\n{self.df.dtypes}\n")
        print(f"Missing Values:\n{self.df.isnull().sum()}\n")
        print(f"Basic Statistics:\n{self.df.describe()}\n")
        print(f"Categorical Description:\n{self.df.describe(include=['category','object'])}\n")
        print("=== END OF DATA SUMMARY ===\n")

    # 2. UNIVARIATE ANALYSIS
    def univariate_analysis(self):
        """Perform univariate analysis on numerical and categorical features."""
        print("=== UNIVARIATE ANALYSIS ===\n")
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(include=['category','object']).columns

        for col in num_cols:
            plt.figure(figsize=(6,4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

        for col in cat_cols:
            plt.figure(figsize=(6,4))
            self.df[col].value_counts().plot(kind="bar")
            plt.title(f"Count Plot of {col}")
            plt.show()
        print("=== END OF UNIVARIATE ANALYSIS ===\n")

    # 3. BIVARIATE ANALYSIS
    def bivariate_analysis(self):
        """Perform bivariate analysis between numerical and categorical features."""
        print("=== BIVARIATE ANALYSIS ===\n")
        # Correlation matrix
        plt.figure(figsize=(10,6))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap="Blues")
        plt.title("Correlation Heatmap")
        plt.show()

        # Scatter plot: TotalPremium vs TotalClaims by ZipCode
        if "PostalCode" in self.df.columns:
            plt.figure(figsize=(8,6))
            sns.scatterplot(data=self.df, x="TotalPremium", y="TotalClaims", hue="PostalCode", legend=False)
            plt.title("Total Premium vs Total Claims by PostalCode")
            plt.show()
        print("=== END OF BIVARIATE ANALYSIS ===\n")

    # 4. OUTLIER DETECTION
    def outlier_detection(self):
        """Detect outliers in numerical features using boxplots."""
        print("=== OUTLIER DETECTION ===\n")
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()
        print("=== END OF OUTLIER DETECTION ===\n")

    # 5. CREATIVE INSIGHT PLOTS
    def creative_insight_plots(self):
        """Generate creative plots to gain insights."""
        print("=== CREATIVE INSIGHT PLOTS ===\n")
        # Insight 1: Loss Ratio by Province
        if "Province" in self.df.columns and "LossRatio" in self.df.columns:
            plt.figure(figsize=(10,6))
            sns.boxplot(data=self.df, x="Province", y="LossRatio")
            plt.title("Loss Ratio by Province")
            plt.show()

        # Insight 2: Claims Severity by Vehicle Type
        if "VehicleType" in self.df.columns and "TotalClaims" in self.df.columns:
            plt.figure(figsize=(10,6))
            sns.violinplot(data=self.df, x="VehicleType", y="TotalClaims")
            plt.title("Claims Severity by Vehicle Type")
            plt.show()

        # Insight 3: Monthly Trends
        if "TransactionMonth" in self.df.columns and "TotalClaims" in self.df.columns:
            self.df["Month"] = self.df["TransactionMonth"].dt.month
            monthly_claims = self.df.groupby("Month")["TotalClaims"].sum().sort_index()
            plt.figure(figsize=(8,6))
            monthly_claims.plot(kind="line", marker="o")
            plt.title("Monthly Trends of Total Claims")
            plt.xlabel("Month")
            plt.ylabel("Total Claims")
            plt.grid(True)
            plt.show()
        print("=== END OF CREATIVE INSIGHT PLOTS ===\n")

    # 6. RUN ALL EDA
    def run_all_eda(self):
        """Run all EDA methods sequentially."""
        self.data_summary()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.outlier_detection()
        self.creative_insight_plots()
        print("=== ALL EDA COMPLETED ===\n")