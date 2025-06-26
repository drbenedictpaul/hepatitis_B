# Step 1.2: Load packages and data
using DataFrames, CSV, Statistics

# Load the dataset
# Make sure the file "AHB_data.csv" is in the same folder
df = CSV.read("AHB_data.csv", DataFrame)

# println("Successfully loaded data. Size: ", size(df))
# println("First 5 rows:")
# println(first(df, 5))

# Step 1.3: Create the 'Outcome' target variable
# 0 = Resolved, 1 = Chronic

# First, create a new column and initialize with a placeholder (missing)
df.Outcome = missings(Int, nrow(df))

# Loop through each row to apply our logic
for row in eachrow(df)
    # Check for follow-up data
    if row."Follow up" == "Yes"
        # Check HBsAg status
        if row.HBsAg == "Neg"
            row.Outcome = 0 # Resolved
        elseif row.HBsAg in ["Pos", "Low Pos"]
            # We only label as chronic if follow-up is >= 6 months
            if !ismissing(row.Months) && row.Months >= 6
                row.Outcome = 1 # Chronic
            end
        end
    end
end

# Now, create a new DataFrame with only the patients we can use for the model
# These are the patients with a known outcome (0 or 1)
model_df = filter(row -> !ismissing(row.Outcome), df)

# println("\nNumber of patients with known outcome: ", nrow(model_df))
# println("Breakdown of outcomes:")
# println(combine(groupby(model_df, :Outcome), nrow))