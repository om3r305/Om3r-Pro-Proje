import json

p = "config_live.json"
with open(p,"r",encoding="utf-8") as f:
    j = json.load(f)

j.setdefault("candles",{})["exit_on_bearish"] = True

with open(p,"w",encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False, indent=2)

print("Güncellendi:", j["candles"]["exit_on_bearish"])
