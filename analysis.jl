# Load necessary packages
using DataFrames, CSV, Statistics, MLJ, CategoricalArrays, ScientificTypes, XGBoost, MLJXGBoostInterface

# --- Part 1: Data Loading and Column Name Cleaning ---
println("Step 1: Loading data and cleaning column names...")
df = CSV.read("AHB_data.csv", DataFrame)

# THIS IS THE DEFINITIVE FIX: Use `\s` to match ANY whitespace character.
# This is more robust than just matching a literal space ' '.
rename!(df, names(df) .=> replace.(names(df), r"[\s\/()-]+" => "_"))
println("-> Cleaned column headers successfully.")

# --- Part 2: Create Outcome and Curate Final Data ---
println("\nStep 2: Creating Outcome variable and curating final data...")
df.Outcome = missings(Union{Int, Missing}, nrow(df))
for r in eachrow(df)
    if r.Follow_up == "Yes"
        if r.HBsAg == "Neg"
            r.Outcome = 0
        elseif r.HBsAg in ["Pos", "Low Pos"] && !ismissing(r.Months) && r.Months >= 6
            r.Outcome = 1
        end
    end
end
final_df = dropmissing(df, :Outcome)
println("-> Found $(nrow(final_df)) patients with known outcomes.")

# --- Part 3: Separating Features (X) and Target (y) ---
println("\nStep 3: Separating features and target...")
y = final_df.Outcome
# The list of columns to drop uses the new, clean names
cols_to_drop = ["No", "Study_ID", "Follow_up", "HBsAg", "Repeat_HBsAg", "Months", "Outcome", "Mutations_RT_domain", "Mutations_SHB_protein", "Mutations_in_Pre_Core_region", "Mutations_in_the_Core_region", "Genotype_AHB"]
X = DataFrames.select(final_df, Not(cols_to_drop))

# --- Part 4: Manual Data Cleaning and Preprocessing ---
println("\nStep 4: Cleaning and preprocessing data...")
# Use the new, clean column name :HBsAg_S_Co
X[!, :HBsAg_S_Co] = map(X[!, :HBsAg_S_Co]) do val
    if ismissing(val) return missing end
    if val == ">250 IU/ml" return 251.0 end
    if val == "neg" return 0.0 end
    return parse(Float64, string(val))
end
# Use the new, clean column names in the coerce! command
coerce!(X, :Gender_AHB => Multiclass, :HBeAg_AHB => Multiclass, :IgM_HAV_AHB => Multiclass, :IgM_HEV_AHB => Multiclass, :Mortality_AHB => Multiclass, :Treatment_AHB => Multiclass)
FillImputer = @load FillImputer pkg=MLJModels; Standardizer = @load Standardizer pkg=MLJModels; OneHotEncoder = @load OneHotEncoder pkg=MLJModels
preprocessor = FillImputer() |> Standardizer() |> OneHotEncoder
mach = machine(preprocessor, X); fit!(mach); X_clean = MLJ.transform(mach, X)
println("-> Preprocessing complete.")

# --- FINAL ANALYSIS PART A: XGBoost Performance Evaluation ---
println("\nFinal Analysis Part A: Evaluating XGBoost Model Performance...")
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost
xgb_model = XGBoostClassifier()
cv = CV(nfolds=nrow(X_clean)) # Leave-One-Out Cross-Validation
y_categorical = categorical(y, ordered=true)

println("-> Evaluating model using cross-validation...")
evaluation = evaluate(xgb_model, X_clean, y_categorical,
                      resampling=cv,
                      measure=[accuracy, auc, f1score])

println("\n--- MODEL PERFORMANCE RESULTS ---")
println(evaluation)


# --- FINAL ANALYSIS PART B: XGBoost Feature Importance ---
println("\nFinal Analysis Part B: Extracting Feature Importances...")
xgb_mach = machine(xgb_model, X_clean, y_categorical)
fit!(xgb_mach, verbosity=0) # verbosity=0 hides the long training logs

importances = feature_importances(xgb_mach)
importance_df = DataFrame(Feature = [p[1] for p in importances], Importance = [p[2] for p in importances])
sort!(importance_df, :Importance, rev=true)
important_features_df = filter(row -> row.Importance > 0.0, importance_df)

println("\n--- FEATURE IMPORTANCE RESULTS ---")
println("The model identified the following features as the most important predictors:")
println(important_features_df)

# Save the final, meaningful result to a CSV file
CSV.write("xgboost_important_features.csv", important_features_df)
println("\n-> Successfully saved importance results to 'xgboost_important_features.csv'.")