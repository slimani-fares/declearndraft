import pstats
import matplotlib.pyplot as plt

# getting the stats dicitonnary 

stats = pstats.Stats("output.prof")

functions = []
for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
    if tt == 0:
        continue
    functions.append({
        "name": name,
        "file": file,
        "tottime": tt,
        "ncalls": nc,
        "percall": tt / nc if nc > 0 else 0,
    })

functions.sort(key=lambda x: x["ncalls"], reverse=True)


# for f in functions[:5]:
#     print(f"{f['tottime']:.3f}s  {f['ncalls']} calls  {f['name']}")



top = functions[:20]
top.reverse()  # reverse so the biggest is at the top of the chart

names = [f["name"] for f in top]
times = [f["tottime"] for f in top]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(names, times)
ax.set_xlabel("tottime (s)")
ax.set_title("Top 20 functions by tottime")
plt.tight_layout()
plt.savefig("profile_tottime.png", dpi=150)
plt.show()