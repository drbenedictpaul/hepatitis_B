# Load necessary packages
using DataFrames, CSV, Plots, StatsPlots, Statistics

# --- Part 1: Load and Prepare Both Datasets ---
println("Step 1: Loading and preparing both AHB and CHABE datasets...")
ahb_df = CSV.read("curated_data.csv", DataFrame)
chabe_df = CSV.read("CHABE_data.csv", DataFrame)
println("-> Loaded $(nrow(ahb_df)) AHB records and $(nrow(chabe_df)) CHABE records.")

# --- Part 2: Create a Unified Grouping Column ---
println("\nStep 2: Creating a 'Group' column for comparison...")
ahb_df.Group = map(o -> o == 0 ? "Acute-Resolved" : "Acute-to-Chronic", ahb_df.Outcome)
chabe_df.Group .= "Established-Chronic"
println("-> Created 'Group' column in both dataframes.")

# --- Part 3: A Robust Method to Combine DataFrames ---
println("\nStep 3: Robustly combining dataframes...")
ahb_plot_df = DataFrame(
    Group = ahb_df.Group,
    ViralLoad = ahb_df.Log_DNA_IU_mL_AHB,
    ALT = ahb_df.SGPT_ALT_AHB,
    IgM = ahb_df[!, :ARC_IgM_Core_S_Co__AHB],
    TB = ahb_df.TB_AHB
)
chabe_plot_df = DataFrame(
    Group = chabe_df.Group,
    ViralLoad = chabe_df[!, :"Log DNA IU/mL_CHBAE"],
    ALT = chabe_df[!, :"SGPT/ALT_CHBAE"],
    IgM = chabe_df[!, :"Architect  IgM Core (S/Co)_CHBAE"],
    TB = chabe_df[!, :"TB_CHBAE"]
)
plot_df = vcat(ahb_plot_df, chabe_plot_df)
println("-> Combined data into a plotting-ready dataframe with $(nrow(plot_df)) records.")

# Save this final, clean dataframe
CSV.write("comparative_plot_data.csv", plot_df)
println("-> Saved the combined plotting data to 'comparative_plot_data.csv'.")


# --- Part 4: Final Cleaning for Plotting ---
println("\nStep 4: Final cleaning for plotting...")
# Final cleaning step: ensure all plotting columns are numeric and drop missing rows
for col in [:ViralLoad, :ALT, :IgM, :TB]
    plot_df[!, col] = allowmissing(plot_df[!, col])
    plot_df[!, col] = map(x -> isa(x, Number) ? x : missing, plot_df[!, col])
    plot_df[!, col] = passmissing(Float64).(plot_df[!, col])
end
dropmissing!(plot_df)
println("-> Final usable records for plotting: $(nrow(plot_df)).")

# --- Step 5: Create the Comparative Box Plots ---
println("\nStep 5: Creating the comparative visualizations...")
gr() # Set plotting backend

# Plot 1: Viral Load
@df plot_df boxplot(:Group, :ViralLoad, group=:Group, legend=false, title="Comparison of Viral Load", ylabel="Log DNA (IU/mL)", xrotation=15)
savefig("comparative_viral_load_boxplot.png")

# Plot 2: ALT Levels
@df plot_df boxplot(:Group, :ALT, group=:Group, legend=false, title="Comparison of SGPT/ALT", ylabel="SGPT/ALT (U/L)", xrotation=15)
savefig("comparative_alt_boxplot.png")

# Plot 3: IgM Levels
@df plot_df boxplot(:Group, :IgM, group=:Group, legend=false, title="Comparison of IgM Core", ylabel="IgM Core (S/Co)", xrotation=15)
savefig("comparative_igm_boxplot.png")

# Plot 4: Total Bilirubin
@df plot_df boxplot(:Group, :TB, group=:Group, legend=false, title="Comparison of Total Bilirubin", ylabel="TB (mg/dL)", xrotation=15)
savefig("comparative_tb_boxplot.png")

println("-> Successfully created and saved all comparative plots.")