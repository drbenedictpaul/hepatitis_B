# Load necessary packages
using DataFrames, CSV, Statistics, MLJ, CategoricalArrays, ScientificTypes, XGBoost, MLJXGBoostInterface

# --- Part 1, 2, 3: Data Loading and Preprocessing (Unchanged) ---
println("Steps 1-3: Loading and preprocessing data...")
df = CSV.read("AHB_data.csv", DataFrame, types=Dict(Symbol("HBsAg S/Co") => String)); df.Outcome = missings(Union{Int, Missing}, nrow(df)); for r in eachrow(df) if r."Follow up" == "Yes" if r.HBsAg == "Neg" r.Outcome = 0 elseif r.HBsAg in ["Pos", "Low Pos"] && !ismissing(r.Months) && r.Months >= 6 r.Outcome = 1 end end end
final_df = dropmissing(df, :Outcome); y = final_df.Outcome; cols_to_drop = ["No", "Study ID", "Follow up", "HBsAg", "Repeat HBsAg", "Months", "Outcome", "Mutations RT domain", "Mutations SHB protein", "Mutations in Pre Core region", "Mutations in the Core region", "Genotype_AHB"]; X = DataFrames.select(final_df, Not(cols_to_drop))
X[!, Symbol("HBsAg S/Co")] = map(X[!, Symbol("HBsAg S/Co")]) do val; if ismissing(val) return missing end; if val == ">250 IU/ml" return 251.0 end; if val == "neg" return 0.0 end; return parse(Float64, val); end
coerce!(X, :Gender_AHB => Multiclass, :HBeAg_AHB => Multiclass, Symbol("IgM HAV_AHB") => Multiclass, Symbol("IgM HEV_AHB") => Multiclass, :Mortality_AHB => Multiclass, :Treatment_AHB => Multiclass)
FillImputer = @load FillImputer pkg=MLJModels; Standardizer = @load Standardizer pkg=MLJModels; OneHotEncoder = @load OneHotEncoder pkg=MLJModels
preprocessor = FillImputer() |> Standardizer() |> OneHotEncoder; mach = machine(preprocessor, X); fit!(mach); X_clean = MLJ.transform(mach, X)
println("-> Preprocessing complete. Final data has $(nrow(X_clean)) samples.")

# --- FINAL ANALYSIS: Using XGBoost for Feature Importance ---
println("\nFinal Analysis: Using XGBoost to find the most important predictors...")
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost

# Instantiate the XGBoost model. The defaults are robust.
xgb_model = XGBoostClassifier()

# Train the model on all the data
y_categorical = categorical(y, ordered=true)
xgb_mach = machine(xgb_model, X_clean, y_categorical)
fit!(xgb_mach)

# Extract feature importances from the trained XGBoost machine
importances = feature_importances(xgb_mach)

# Create and sort the results DataFrame
importance_df = DataFrame(Feature = [p[1] for p in importances], Importance = [p[2] for p in importances])
sort!(importance_df, :Importance, rev=true)

# Filter to show only features with non-zero importance
important_features_df = filter(row -> row.Importance > 0.0, importance_df)


# --- FINAL RESULTS ---
println("\n--- XGBOOST FEATURE IMPORTANCE RESULTS ---")
println("The model identified the following features as the most important predictors:")
println(important_features_df)

# Save the final, meaningful result to a CSV file
CSV.write("xgboost_important_features.csv", important_features_df)
println("\n-> Successfully saved final results to 'xgboost_important_features.csv'.")