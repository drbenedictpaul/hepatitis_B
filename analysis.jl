# Load necessary packages
using DataFrames, CSV, Statistics, MLJ, CategoricalArrays, ScientificTypes

# --- Part 1: Data Loading and Label Creation ---
println("Step 1: Loading data and creating Outcome variable...")
df = CSV.read("AHB_data.csv", DataFrame, types=Dict(Symbol("HBsAg S/Co") => String))
df.Outcome = missings(Union{Int, Missing}, nrow(df))
for r in eachrow(df)
    if r."Follow up" == "Yes"
        if r.HBsAg == "Neg"
            r.Outcome = 0
        elseif r.HBsAg in ["Pos", "Low Pos"] && !ismissing(r.Months) && r.Months >= 6
            r.Outcome = 1
        end
    end
end

# --- Part 2: Final Data Curation ---
println("\nStep 2: Final data curation...")
final_df = dropmissing(df, :Outcome)
println("-> Final model-ready data has $(nrow(final_df)) samples.")
y = final_df.Outcome
cols_to_drop = ["No", "Study ID", "Follow up", "HBsAg", "Repeat HBsAg", "Months", "Outcome", "Mutations RT domain", "Mutations SHB protein", "Mutations in Pre Core region", "Mutations in the Core region", "Genotype_AHB"]
X = DataFrames.select(final_df, Not(cols_to_drop))

# --- Step 3: Manual Data Cleaning and Preprocessing ---
println("\nStep 3: Manual data cleaning and preprocessing...")
X[!, Symbol("HBsAg S/Co")] = map(X[!, Symbol("HBsAg S/Co")]) do val
    if ismissing(val) return missing end
    if val == ">250 IU/ml" return 251.0 end
    if val == "neg" return 0.0 end
    return parse(Float64, val)
end
coerce!(X, :Gender_AHB => Multiclass, :HBeAg_AHB => Multiclass, Symbol("IgM HAV_AHB") => Multiclass, Symbol("IgM HEV_AHB") => Multiclass, :Mortality_AHB => Multiclass, :Treatment_AHB => Multiclass)

FillImputer = @load FillImputer pkg=MLJModels
Standardizer = @load Standardizer pkg=MLJModels
OneHotEncoder = @load OneHotEncoder pkg=MLJModels
preprocessor = FillImputer() |> Standardizer() |> OneHotEncoder()
mach = machine(preprocessor, X)
fit!(mach)
X_clean = MLJ.transform(mach, X)
println("-> Preprocessing complete.")

# --- Step 4: Saving the Cleaned Data ---
println("\nStep 4: Saving the cleaned data...")
X_clean_df = X_clean |> DataFrame
y_df = DataFrame(Outcome = categorical(y))
CSV.write("X_clean.csv", X_clean_df)
CSV.write("y_clean.csv", y_df)
println("-> Successfully created 'X_clean.csv' and 'y_clean.csv'.")

# --- Step 5: Training and Evaluating a Random Forest Model ---
println("\nStep 5: Training and evaluating the model...")
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
rf_model = RandomForestClassifier()
cv = CV(nfolds=nrow(X_clean))

# THIS IS THE FIX: We add `ordered=true` to satisfy the f1score metric.
y_categorical = categorical(y, ordered=true)

println("-> Evaluating model using cross-validation...")
evaluation = evaluate(rf_model, X_clean, y_categorical,
                      resampling=cv,
                      measure=[accuracy, auc, f1score])

println("\n--- MODEL EVALUATION COMPLETE ---")
println("Cross-validation results for the Random Forest model:")
println(evaluation)

# --- Step 6: Extract and Analyze Feature Importances ---
println("\nStep 6: Extracting Feature Importances (The Key Result)...")

# Train a final model on all data for interpretation
println("-> Training a final model on all data...")
final_mach = machine(rf_model, X_clean, y_categorical)
fit!(final_mach)

# Get the feature importances from the trained machine
importances = feature_importances(final_mach)

# THIS IS THE FIX: We construct the DataFrame from the list of pairs.
# This is a more robust way to handle the output.
importance_df = DataFrame(
    Feature = [pair[1] for pair in importances],
    Importance = [pair[2] for pair in importances]
)

# Sort the DataFrame to see the most important features at the top.
sort!(importance_df, :Importance, rev=true)


# --- FINAL RESULT ---
println("\n--- FEATURE IMPORTANCE RESULTS ---")
println("The clinical factors ranked by their importance for predicting the outcome:")
println(importance_df)