# from src.preprocess import preprocess
# from src.visualize import plot_heatmap, plot_gender_married_status
# from src.train import split_data, train_svm
# from src.evaluate import evaluate

def main():
    # Step 1: Load & Preprocess
    # Use the data variable that was already loaded in the notebook
    processed_data = preprocess("/LoanApprovalPrediction.csv")

    # Step 2: Visualize
    plot_heatmap(processed_data)
    plot_gender_married_status(processed_data)

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = split_data(processed_data)

    # Step 4: Train Model
    model = train_svm(X_train, y_train)

    # Step 5: Evaluate
    evaluate(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
