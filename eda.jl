# Load necessary packages for EDA
using DataFrames, CSV, Statistics, Plots, StatsPlots

# Set the plotting backend
gr()

# --- Part 1: Load Curated Data ---
# println("Step 1: Loading curated data for analysis...")
df = CSV.read("curated_data.csv", DataFrame)
# println("-> Successfully loaded $(nrow(df)) patient records.")


# --- Part 2: Descriptive Statistics ---
println("\nStep 2: Generating and saving descriptive statistics...")
desc_stats = describe(df, :mean, :std, :median, :min, :max)
CSV.write("descriptive_statistics.csv", desc_stats, transform=(col, val) -> something(val, ""))
println("\n--- DESCRIPTIVE STATISTICS TABLE ---"); println(desc_stats)
# println("\n-> Successfully saved descriptive statistics to 'descriptive_statistics.csv'.")


# --- Part 3: Visualize Feature #1 (ALT) ---
println("\nStep 3: Creating visualization for SGPT/ALT...")
@df df boxplot(:Outcome, :SGPT_ALT_AHB, group=:Outcome, legend=false, title="Initial SGPT/ALT Levels by Patient Outcome", xlabel="Final Outcome", ylabel="SGPT/ALT (U/L)"); xticks!([0, 1], ["Resolved", "Chronic"]); savefig("alt_boxplot.png")
println("-> Successfully saved 'alt_boxplot.png'.")


# --- Part 4: Visualize Feature #2 (IgM) ---
println("\nStep 4: Creating visualization for IgM Core...")
@df df boxplot(:Outcome, :ARC_IgM_Core_S_Co__AHB, group=:Outcome, legend=false, title="Initial IgM Core Levels by Patient Outcome", xlabel="Final Outcome", ylabel="IgM Core (S/Co)"); xticks!([0, 1], ["Resolved", "Chronic"]); savefig("igm_boxplot.png")
println("-> Successfully saved box plot to 'igm_boxplot.png'.")


# --- Part 5: Visualize Continuous vs. Continuous (Scatter Plot) ---
println("\nStep 5: Creating Scatter Plot (ALT vs. Total Bilirubin)...")
@df df scatter(:SGPT_ALT_AHB, :TB_AHB, group = :Outcome, title = "ALT vs. Total Bilirubin at Diagnosis", xlabel = "SGPT/ALT (U/L)", ylabel = "Total Bilirubin (mg/dL)", legend = :topleft)
savefig("alt_vs_tb_scatter.png")
println("-> Successfully saved scatter plot to 'alt_vs_tb_scatter.png'.")


# --- Part 6: Visualize Categorical vs. Categorical (Bar Chart) - ROBUST METHOD ---
println("\nStep 6: Creating Grouped Bar Chart (Outcome by Treatment)...")

# THIS IS THE FIX: We pre-process the data for the bar chart.
# 1. Fill any missing treatment values with a placeholder like "Unknown"
df.Treatment_AHB = coalesce.(df.Treatment_AHB, "Unknown")

# 2. Group the data by Treatment and Outcome and count the patients in each group
bar_data = combine(groupby(df, [:Treatment_AHB, :Outcome]), nrow => :Count)

# 3. Create the grouped bar chart from this pre-summarized data
@df bar_data groupedbar(
    :Treatment_AHB,
    :Count,
    group = :Outcome,
    title = "Patient Outcome by Initial Treatment Status",
    xlabel = "Initial Treatment",
    ylabel = "Number of Patients"
)
savefig("outcome_by_treatment_barchart.png")
println("-> Successfully saved bar chart to 'outcome_by_treatment_barchart.png'.")