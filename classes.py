import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("Data/ground_truth_labels.csv")
df.columns = df.columns.str.strip().str.replace('\ufeff', '')

cols = ['Moderate or greater MS', 'Moderate or greater MR',
        'Moderate or greater AR', 'Moderate or greater AS',
        'Moderate or greater TR']
short = ['MS', 'MR', 'AR', 'AS', 'TR']
df.columns = [c.strip() for c in df.columns]

df['total'] = df[cols].sum(axis=1)
df['has_AS'] = df['Moderate or greater AS'] == 1

singular, as_comorbid, other, none_group = [], [], [], []

for _, row in df.iterrows():
    t = int(row['total'])
    if t == 0:
        none_group.append(row)
    elif t == 1:
        singular.append(row)
    elif t == 2 and row['has_AS']:
        as_comorbid.append(row)
    else:
        other.append(row)

singular_df = pd.DataFrame(singular)
as_df = pd.DataFrame(as_comorbid)
other_df = pd.DataFrame(other)

singular_counts = {s: int((singular_df[c] == 1).sum()) if len(singular_df) else 0
                   for s, c in zip(short, cols)}

as_counts = {}
for s, c in zip(short, cols):
    if s == 'AS':
        continue
    as_counts[f'AS+{s}'] = int((as_df[c] == 1).sum()) if len(as_df) else 0

rest_combos = {}
if len(other_df):
    for _, row in other_df.iterrows():
        combo = '+'.join([s for s, c in zip(short, cols) if row[c] == 1])
        rest_combos[combo] = rest_combos.get(combo, 0) + 1
rest_combos = dict(sorted(rest_combos.items(), key=lambda x: -x[1]))

colors_singular = ['#378ADD', '#3C3489', '#1D9E75', '#D85A30', '#639922']
colors_as = ['#378ADD', '#3C3489', '#1D9E75', '#639922']

def add_value_labels_v(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                    str(int(h)), ha='center', va='bottom', fontsize=11, fontweight='500')

def add_value_labels_h(ax, bars):
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(int(w)), ha='left', va='center', fontsize=10, fontweight='500')

# --- Plot 1: Singular ---
fig1, ax1 = plt.subplots(figsize=(12, 6))
fig1.patch.set_facecolor('#ffffff')
ax1.set_facecolor('#ffffff')
labels1 = list(singular_counts.keys())
vals1 = list(singular_counts.values())
bars1 = ax1.bar(labels1, vals1, color=colors_singular, edgecolor='none', width=0.5)
for bar in bars1:
    bar.set_linewidth(0)
    bar.set_capstyle('round') if hasattr(bar, 'set_capstyle') else None
add_value_labels_v(ax1, bars1)
ax1.set_title('Singular disease patients (exactly 1 condition)', fontsize=13, fontweight='500', pad=12)
ax1.set_ylabel('Patients', fontsize=11)
ax1.set_ylim(0, max(vals1) + 3)
ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color('#cccccc')
ax1.tick_params(colors='#555555')
ax1.grid(axis='y', color='#eeeeee', linewidth=0.8)
ax1.set_axisbelow(True)

legend1 = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_singular, labels1)]
ax1.legend(handles=legend1, fontsize=10, frameon=False, loc='upper right')
fig1.tight_layout(pad=2.0)

# --- Plot 2: AS comorbidity ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')
labels2 = list(as_counts.keys())
vals2 = list(as_counts.values())
bars2 = ax2.bar(labels2, vals2, color=colors_as, edgecolor='none', width=0.5)
add_value_labels_v(ax2, bars2)
ax2.set_title('AS comorbidity — patients with AS + exactly 1 other condition', fontsize=13, fontweight='500', pad=12)
ax2.set_ylabel('Patients', fontsize=11)
ax2.set_ylim(0, max(vals2) + 3)
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color('#cccccc')
ax2.tick_params(colors='#555555')
ax2.grid(axis='y', color='#eeeeee', linewidth=0.8)
ax2.set_axisbelow(True)

legend2 = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_as, labels2)]
ax2.legend(handles=legend2, fontsize=10, frameon=False, loc='upper right')
fig2.tight_layout(pad=2.0)

# --- Plot 3: Everything else (horizontal) ---
fig3, ax3 = plt.subplots(figsize=(12, 8))
fig3.patch.set_facecolor('#ffffff')
ax3.set_facecolor('#ffffff')
labels3 = list(rest_combos.keys())
vals3 = list(rest_combos.values())
colors3 = ['#444441' if l.count('+') >= 2 else '#888780' for l in labels3]
y_pos = np.arange(len(labels3))
bars3 = ax3.barh(y_pos, vals3, color=colors3, edgecolor='none', height=0.6)
add_value_labels_h(ax3, bars3)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels3, fontsize=10)
ax3.invert_yaxis()
ax3.set_title('Everything else — all other multi-condition combinations', fontsize=13, fontweight='500', pad=12)
ax3.set_xlabel('Patients', fontsize=11)
ax3.set_xlim(0, max(vals3) + 2)
ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax3.spines[['top', 'right']].set_visible(False)
ax3.spines[['left', 'bottom']].set_color('#cccccc')
ax3.tick_params(colors='#555555')
ax3.grid(axis='x', color='#eeeeee', linewidth=0.8)
ax3.set_axisbelow(True)

legend3 = [
    mpatches.Patch(color='#888780', label='2 conditions (non-AS pairs)'),
    mpatches.Patch(color='#444441', label='3+ conditions'),
]
ax3.legend(handles=legend3, fontsize=10, frameon=False, loc='lower right')

fig3.tight_layout(pad=2.0)
plt.show()