import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11

df = pd.read_csv('1.csv')

plt.figure(figsize=(10, 6))

colors = {'MulVul': '#E74C3C', 'LLMxCPG': '#3498DB', 'LLMVulExp': '#2ECC71', 'VISION': '#9B59B6'}
markers = {'MulVul': 'o', 'LLMxCPG': 's', 'LLMVulExp': '^', 'VISION': 'd'}

for method in ['VISION', 'LLMVulExp', 'LLMxCPG', 'MulVul']:
    vals = df[method].astype(float) * 100
    plt.scatter(df['Support'], vals, alpha=0.6,
                label=f'{method} (Macro F1={vals.mean():.1f}%)',
                c=colors[method], marker=markers[method], s=40, edgecolors='white', linewidth=0.5)

plt.xscale('log')
plt.xlabel('Sample count per CWE (log scale)', fontsize=36)
plt.ylabel('F1 Score (%)', fontsize=36)
# plt.title('CWE-level F1 Score vs Support (Primevul Dataset)', fontsize=12)
plt.legend(loc='upper left', fontsize=20, frameon=True, edgecolor='none')
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.tick_params(axis='both', labelsize=28)
plt.tight_layout()
plt.savefig('1.png', dpi=300)
print('Saved to 1.png')
