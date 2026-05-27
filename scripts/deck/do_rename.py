import os, re
files = ['investor_deck_mt_lnn.tex', 'investor_deck_mt_lnn_zh.tex', 'AWARENESS_NETWORK_PRD.md', 'plot_awareness_network.py', 'README.md']
for filepath in files:
  if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
    content = content.replace('M1 架构', 'AwareLiquid 架构')
    content = content.replace('M1 (MT-LNN)', 'AwareLiquid (MT-LNN)')
    content = content.replace('Local Brain (Edge M1)', 'Local Brain (AwareLiquid)')
    content = content.replace('Cloud Oracle (Awareness Cloud)', 'Awareness Cloud')
    content = content.replace('本地大脑 (Edge M1)', '本地大脑 (AwareLiquid)')
    content = content.replace('感知云端 (Awareness Cloud)', 'Awareness Cloud')
    content = re.sub(r'(?<=[\s，。、：；（])M1(?=[\s，。、：；）\.\,\!\?])', 'AwareLiquid', content)
    content = re.sub(r'(\b)M1(?=[\s，。、：；）\.\,\!\?])', r'\g<1>AwareLiquid', content)
    content = content.replace('github.com/everest-an/AwareLiquid', 'github.com/everest-an/M1')
    with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
    print('Updated', filepath)
