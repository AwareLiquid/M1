with open('e:/M1/plot_recent_updates.py', 'r', encoding='utf-8') as f:
    text = f.read()

# remove null bytes
text = text.replace('\x00', '').replace('\n\np', '')

with open('e:/M1/plot_recent_updates.py', 'w', encoding='utf-8') as f:
    f.flush()
    f.write(text + "\nplt.savefig('fig_operator_compression_updates.pdf', dpi=300)\n")
