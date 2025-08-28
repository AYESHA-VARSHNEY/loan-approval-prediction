import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical(df):
    """Bar plots of categorical columns"""
    obj_cols = df.select_dtypes(include=['object']).columns
    plt.figure(figsize=(18,36))
    index = 1
    for col in obj_cols:
        y = df[col].value_counts()
        plt.subplot(11,4,index)
        plt.xticks(rotation=90)
        sns.barplot(x=list(y.index), y=y)
        index += 1
    plt.show()

def plot_heatmap(df):
    """Heatmap of correlations"""
    plt.figure(figsize=(12,6))
    sns.heatmap(df.corr(), cmap='BrBG', fmt='.2f',
                linewidths=2, annot=True)
    plt.show()

def plot_gender_married_status(df):
    """Relation between Gender, Married and Loan_Status"""
    sns.catplot(x="Gender", y="Married",
                hue="Loan_Status", kind="bar", data=df)
    plt.show()
