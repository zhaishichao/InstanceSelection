import matplotlib.pyplot as plt
import numpy as np

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical symbols

# Data
y1 = [0.83378, 0.85539, 0.86606, 0.870891, 0.87417, 0.87886, 0.87955, 0.88023, 0.88023, 0.88023, 0.88023]
y2 = [0.87743, 0.8884, 0.88839, 0.88963, 0.89067, 0.89125, 0.8923, 0.8923, 0.8923, 0.8923, 0.8923]
x = np.arange(0, len(y1)*3,step=3)  # Runs from 0 to 10

# Create figure with tight layout
fig, ax = plt.subplots(figsize=(8, 7.2))

# Set axis limits to start from 0 (with slight padding)
ax.set_xlim(-0.5, 30.5)
ax.set_ylim(0.82, 0.90)

# Configure grid (behind other elements, dashed lines)
ax.grid(True, linestyle='--', alpha=0.6, which='both')
ax.set_axisbelow(True)  # Grid behind data

# Set ticks with consistent spacing
ax.set_xticks([0,3,6,9,12,15,18,21,24,27,30])
# ax.set_xticklabels(np.arange(0, len(y1)))  # Run numbers start at 1
ax.set_yticks(np.arange(0.83, 0.91, 0.01))

# Plot lines with markers (starting at x=0)
line1, = ax.plot(x, y1, 'o-', color='#1f77b4', markersize=8, linewidth=1.5, label='E-MOSAIC',zorder=100)
line2, = ax.plot(x, y2, 'o-', color='#ff7f0e', markersize=8, linewidth=1.5, label='mile',zorder=100)

# Labels with Times New Roman (size 12)
ax.set_xlabel('Generation', fontsize=14)
ax.set_ylabel('G-mean', fontsize=14)
ax.set_title('WallRobot', fontsize=14, pad=10)

# Legend with Times New Roman (size 11)
ax.legend(fontsize=11, framealpha=1, loc='center',bbox_to_anchor=(0.70, 0.35))
# Adjust layout to prevent clipping
plt.tight_layout()

# Save as high-resolution PNG (optional)
plt.savefig('WallRobot.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1)
plt.show()