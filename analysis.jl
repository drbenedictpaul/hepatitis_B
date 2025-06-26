# Load necessary packages
using DataFrames, CSV, Statistics, MLJ, CategoricalArrays, ScientificTypes

# --- Part 1 & 2: Load Data and Separate X/y ---
println("Step 1 & 2: Loading data, creating Outcome, and separating X/y...")
# Read HBsAg S/Co as String to handle its mixed content
df = CSV.read("AHB_data.csv", DataFrame, types=Dict(Symbol("HBsAg S/Co") => String))
df.Outcome = missings(Int, nrow(df))
for row in eachrow(df)
    if row."Follow up" == "Yes"
        if row.HBsAg == "Neg"
            row.Outcome = 0
        elseif row.HBsAg in ["Pos", "Low Pos"] && !ismissing(row.Months) && row.Months >= 6
            row.Outcome = 1
        end
    end
end
model_df = filter(row -> !ismissing(row.Outcome), df)
y = model_df.Outcome
cols_to_drop = ["No", "Study ID", "Follow up", "HBsAg", "Repeat HBsAg", "Months", "Outcome", "Mutations RT domain", "Mutations SHB protein", "Mutations in Pre Core region", "Mutations in the Core region", "Genotype_AHB"]
X = DataFrames.select(model_df, Not(cols_to_drop))
println("-> Found $(nrow(model_df)) patients. Separated into X and y.")

# --- Step 3: Manual Data Cleaning and Preprocessing ---
println("\nStep 3: Manual data cleaning and preprocessing...")

# Manually clean the 'HBsAg S/Co' column
# THIS IS THE FIX: We add an `elseif` to handle the 'neg' case.
X[!, Symbol("HBsAg S/Co")] = map(X[!, Symbol("HBsAg S/Co")]) do val
    if val == ">250 IU/ml"
        return 251.0
    elseif val == "neg"
        return 0.0 # Treat 'neg' as a numerical zero
    else
        return parse(Float64, val)
    end
end
println("-> Manually cleaned 'HBsAg S/Co' column.")

# Manually coerce the scientific types
coerce!(X, :Gender_AHB => Multiclass)
coerce!(X, :HBeAg_AHB => Multiclass)
coerce!(X, Symbol("IgM HAV_AHB") => Multiclass)
coerce!(X, Symbol("IgM HEV_AHB") => Multiclass)
coerce!(X, :Mortality_AHB => Multiclass)
coerce!(X, :Treatment_AHB => Multiclass)
println("-> Manually coerced categorical columns to Multiclass.")

# Now, build the pipeline
FillImputer = @load FillImputer pkg=MLJModels
Standardizer = @load Standardizer pkg=MLJModels
OneHotEncoder = @load OneHotEncoder pkg=MLJModels

preprocessor = FillImputer() |> Standardizer() |> OneHotEncoder()

mach = machine(preprocessor, X)
fit!(mach)

X_clean = MLJ.transform(mach, X)
y_categorical = categorical(y)
println("-> Preprocessing complete.")


# --- Step 4: Saving the Cleaned Data ---
println("\nStep 4: Saving the cleaned data...")

X_clean_df = X_clean |> DataFrame
y_df = DataFrame(Outcome = y_categorical)

CSV.write("X_clean.csv", X_clean_df)
CSV.write("y_clean.csv", y_df)

println("\n--- ALL DONE ---")
println("Successfully created 'X_clean.csv' and 'y_clean.csv'.")