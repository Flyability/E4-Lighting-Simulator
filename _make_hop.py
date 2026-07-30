import json

src = r'C:\Users\deskt\Downloads\E4_LightingSim (3)\E4_LightingSim\configs\harmony_oris.json'
dst = r'C:\Users\deskt\Downloads\E4_LightingSim (3)\E4_LightingSim\configs\hop.json'

with open(src, 'r') as f:
    cfg = json.load(f)

# Give each non-panel-slot group a unique template_name
# so they won't share a master folder with shared sliders
idx = 1
for g in cfg['custom_groups']:
    if g.get('panel_slot') is None and g.get('template_name'):
        g['template_name'] = f"{g['template_name']}_grp{idx}"
        idx += 1

cfg['name'] = 'hop'

with open(dst, 'w') as f:
    json.dump(cfg, f, indent=4)

print(f'Created {dst}')
print(f'Groups template_names: {[g.get("template_name") for g in cfg["custom_groups"]]}')
