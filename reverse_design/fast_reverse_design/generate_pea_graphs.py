"""
Generate graphs for SEP-CNN, ResNet, and PEA analysis.
Real data from FreiHAND/PEA_Analysis/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
output_dir = Path("FreiHAND/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

# Load real data
sepcnn_times = np.loadtxt("FreiHAND/PEA_Analysis/PEA_design_time/sepcnn_PEA_time_mutate0.55.csv")
resnet_times = np.loadtxt("FreiHAND/PEA_Analysis/PEA_design_time/resnet_PEA_time_mutate0.55.csv")
pea_times = np.loadtxt("FreiHAND/PEA_Analysis/PEA_design_time/PEA_time_mutate0.55.csv")

# Load generation data
gen_data = np.loadtxt("FreiHAND/PEA_Analysis/PEA_total_gen/PEA_total_gen_mutate0.55.csv", delimiter=",")
generations = gen_data[:, 0]
total_evaluations = gen_data[:, 1]

# ============================================================================
# GRAPH 1: Design Time Comparison (SEP-CNN vs ResNet vs PEA)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

runs = np.arange(1, len(sepcnn_times) + 1)

ax.plot(runs, sepcnn_times, marker='o', label='SEP-CNN PEA', linewidth=2, markersize=6)
ax.plot(runs, resnet_times, marker='s', label='ResNet PEA', linewidth=2, markersize=6)
ax.plot(runs, pea_times, marker='^', label='Baseline PEA', linewidth=2, markersize=6)

ax.set_xlabel("Run Number", fontsize=12, fontweight='bold')
ax.set_ylabel("Design Time (seconds)", fontsize=12, fontweight='bold')
ax.set_title("PEA Design Time Comparison: SEP-CNN vs ResNet vs Baseline\n(Mutation Rate = 0.55)", 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xticks(runs)

plt.tight_layout()
plt.savefig(output_dir / "04_pea_design_time_comparison.png", dpi=300, bbox_inches='tight')
print("Saved: {}".format(output_dir / '04_pea_design_time_comparison.png'))
plt.close()

# ============================================================================
# GRAPH 2: Box Plot Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

data_to_plot = [sepcnn_times, resnet_times, pea_times]
bp = ax.boxplot(data_to_plot, labels=['SEP-CNN PEA', 'ResNet PEA', 'Baseline PEA'], patch_artist=True)

# Color the boxes
colors = ['#3498db', '#e74c3c', '#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel("Design Time (seconds)", fontsize=12, fontweight='bold')
ax.set_title("Design Time Distribution Across Runs\n(Mutation Rate = 0.55)", 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add summary statistics
stats_text = (
    "SEP-CNN: mu={:.2f}s, sigma={:.2f}s\n"
    "ResNet:  mu={:.2f}s, sigma={:.2f}s\n"
    "PEA:     mu={:.2f}s, sigma={:.2f}s"
).format(np.mean(sepcnn_times), np.std(sepcnn_times),
         np.mean(resnet_times), np.std(resnet_times),
         np.mean(pea_times), np.std(pea_times))
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / "05_pea_design_time_boxplot.png", dpi=300, bbox_inches='tight')
print("Saved: {}".format(output_dir / '05_pea_design_time_boxplot.png'))
plt.close()

# ============================================================================
# GRAPH 3: Generations vs Total Evaluations
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by generations for clearer visualization
sorted_idx = np.argsort(generations)
gens_sorted = generations[sorted_idx]
evals_sorted = total_evaluations[sorted_idx]

ax.scatter(gens_sorted, evals_sorted, s=100, alpha=0.6, edgecolors='black', linewidth=1)
ax.plot(gens_sorted, evals_sorted, alpha=0.4, linewidth=1, linestyle='--')

ax.set_xlabel("Generations", fontsize=12, fontweight='bold')
ax.set_ylabel("Total Evaluations", fontsize=12, fontweight='bold')
ax.set_title("PEA: Total Evaluations vs Generations\n(Population=2000, Mutation Rate=0.55)", 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "06_pea_generations_evaluations.png", dpi=300, bbox_inches='tight')
print("Saved: {}".format(output_dir / '06_pea_generations_evaluations.png'))
plt.close()

# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS (Real Data)")
print("="*70)
print("\nSEP-CNN PEA Design Time:")
print("  Mean:    {:.3f} s".format(np.mean(sepcnn_times)))
print("  Std Dev: {:.3f} s".format(np.std(sepcnn_times)))
print("  Min:     {:.3f} s".format(np.min(sepcnn_times)))
print("  Max:     {:.3f} s".format(np.max(sepcnn_times)))
print("  Median:  {:.3f} s".format(np.median(sepcnn_times)))

print("\nResNet PEA Design Time:")
print("  Mean:    {:.3f} s".format(np.mean(resnet_times)))
print("  Std Dev: {:.3f} s".format(np.std(resnet_times)))
print("  Min:     {:.3f} s".format(np.min(resnet_times)))
print("  Max:     {:.3f} s".format(np.max(resnet_times)))
print("  Median:  {:.3f} s".format(np.median(resnet_times)))

print("\nBaseline PEA Design Time:")
print("  Mean:    {:.3f} s".format(np.mean(pea_times)))
print("  Std Dev: {:.3f} s".format(np.std(pea_times)))
print("  Min:     {:.3f} s".format(np.min(pea_times)))
print("  Max:     {:.3f} s".format(np.max(pea_times)))
print("  Median:  {:.3f} s".format(np.median(pea_times)))

print("\nGenerations vs Evaluations:")
print("  Generations range:    {} - {}".format(int(np.min(generations)), int(np.max(generations))))
print("  Evaluations range:    {} - {}".format(int(np.min(total_evaluations)), int(np.max(total_evaluations))))
print("  Total runs:           {}".format(len(generations)))

print("\nAll graphs generated successfully!")
print("="*70)
