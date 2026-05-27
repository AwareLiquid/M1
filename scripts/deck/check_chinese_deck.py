import re
with open('e:/M1/investor_deck_mt_lnn_zh.tex', 'r', encoding='utf-8') as f:
    text = f.read()

frames = ['我们的方案：MT-LNN 架构登场', '超越“文字接龙”：真正理解真实物理世界规律', '深度解析：并行扫描与量子耦合']
for frame in frames:
    print('--- ' + frame + ' ---')
    idx = text.find(frame)
    if idx != -1:
        match = re.search(r'\\begin\{frame\}.*?\{.*?\}.*?\\end\{frame\}', text[idx-20:], re.DOTALL)
        if match:
            print(match.group(0))
