import matplotlib.pyplot as plt
import matplotlib.patches as patches

def add_stage(ax, label, x, y, width, height, color):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color, lw=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=10)

# Création de la figure
fig, ax = plt.subplots(figsize=(10, 8))

# Définition des tailles et positions
x_pos = 1
y_pos = 8
stage_height = 0.8
stage_width = 3

# Ajout des blocs au schéma
# Entrée
add_stage(ax, "Entrée\n(3, 256, 256)", x_pos, y_pos, stage_width, stage_height, 'lightblue')

# Stage 1 (Initial Conv + MaxPool)
y_pos -= 1.5
add_stage(ax, "Stage 1\nConv(64) + MaxPool\n(64, 128, 128)", x_pos, y_pos, stage_width, stage_height, 'lightgreen')

# Stage 2
y_pos -= 1.5
add_stage(ax, "Stage 2\nShortcut + 3x Residual\n(256, 128, 128)", x_pos, y_pos, stage_width, stage_height, 'lightcoral')

# Stage 3
y_pos -= 1.5
add_stage(ax, "Stage 3\nShortcut + 4x Residual\n(512, 64, 64)", x_pos, y_pos, stage_width, stage_height, 'lightsalmon')

# Stage 4
y_pos -= 1.5
add_stage(ax, "Stage 4\nShortcut + 6x Residual\n(1024, 32, 32)", x_pos, y_pos, stage_width, stage_height, 'lightpink')

# Average Pooling
y_pos -= 1.5
add_stage(ax, "Average Pooling\n(1024, 4, 4)", x_pos, y_pos, stage_width, stage_height, 'lightgray')

# Classification
y_pos -= 1.5
add_stage(ax, "Fully Connected\n(1024*4*4 → Classes)", x_pos, y_pos, stage_width, stage_height, 'yellow')

# Ajuster les limites
ax.set_xlim(0, 6)
ax.set_ylim(0, 10)
ax.axis('off')

# Affichage du diagramme
plt.title('Architecture ResNet (3, 2)', fontsize=14)
plt.show()