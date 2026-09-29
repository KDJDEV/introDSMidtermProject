"""
639 Bach compositions and 94 Tchaikovsky compositions
"""

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

bach_path = "BachSlices.csv"
tchaikovsky_path = "TchaikovskySlices.csv"

bach = pd.read_csv(bach_path)
tchaikovsky = pd.read_csv(tchaikovsky_path)

print(bach.head().to_string())
bach["composer"] = "Bach"
tchaikovsky["composer"] = "Tchaikovsky"
df = pd.concat([bach, tchaikovsky], ignore_index=True)

def parse_normal_form(x):
    return tuple(ast.literal_eval(x))

df["normal_form"] = df["NormalForm"].apply(parse_normal_form)

piece_stats = (
    df.groupby(["composer", "file"])
      .agg(
          harmonicVocabulary=("normal_form", "nunique"),
          events=("normal_form", "size"),
          avgSonoritySize=("normal_form", lambda x: np.mean([len(v) for v in x])),
          chromaticRate=("normal_form", lambda x: np.mean([
              any(((b - a) % 12 in (1, 11)) for i, a in enumerate(v) for b in v[i+1:])
              for v in x
          ]))
      )
      .reset_index()
)

def entropy(values):
    counts = pd.Series(values).value_counts(normalize=True)
    return -(counts * np.log2(counts)).sum()

entropy_df = (
    df.groupby(["composer", "file"])["normal_form"]
      .apply(entropy)
      .reset_index(name="harmonicEntropy")
)

piece_stats = piece_stats.merge(entropy_df, on=["composer", "file"])

plt.figure(figsize=(8, 5))
for composer in ["Bach", "Tchaikovsky"]:
    vals = piece_stats.loc[piece_stats["composer"] == composer, "harmonicEntropy"]
    plt.hist(vals, bins=25, alpha=0.55, label=composer, density=False)
plt.xlabel("Harmonic vocabulary entropy (bits)")
plt.ylabel("Number of compositions")
plt.title("Harmonic vocabulary diversity: Bach vs. Tchaikovsky")
plt.legend()
plt.savefig("bach_tchaikovsky_harmonic_entropy.png")
#plt.show()

plt.figure(figsize=(8, 5))
data = [
    piece_stats.loc[piece_stats["composer"] == "Bach", "harmonicVocabulary"],
    piece_stats.loc[piece_stats["composer"] == "Tchaikovsky", "harmonicVocabulary"]
]
plt.boxplot(data, labels=["Bach", "Tchaikovsky"], showfliers=False)
plt.ylabel("Distinct pitch-class sets per composition")
plt.title("Harmonic vocabulary size: Bach vs. Tchaikovsky")
plt.savefig("bach_tchaikovsky_harmonic_vocabulary.png")
#plt.show()

counts = df.groupby(["composer", "normal_form"]).size()
freq = (counts / counts.groupby(level="composer").transform("sum")).rename("proportion").reset_index()

wide = freq.pivot(index="normal_form", columns="composer", values="proportion").fillna(0)
wide["difference"] = wide.get("Tchaikovsky", 0) - wide.get("Bach", 0)

top = pd.concat([
    wide.nlargest(8, "difference"),
    wide.nsmallest(8, "difference")
]).sort_values("difference")

labels = [str(x) for x in top.index]
plt.figure(figsize=(10, 7))
plt.barh(labels, top["difference"] * 100)
plt.axvline(0, linewidth=1)
plt.xlabel("Tchaikovsky proportion − Bach proportion (percentage points)")
plt.ylabel("Pitch-class set (NormalForm)")
plt.title("Pitch-class sets that distinguish Tchaikovsky from Bach")

plt.savefig("bach_tchaikovsky_distinctive_sonorities.png")
#plt.show()

plt.figure(figsize=(8, 5))
data = [
    piece_stats.loc[piece_stats["composer"] == "Bach", "chromaticRate"] * 100,
    piece_stats.loc[piece_stats["composer"] == "Tchaikovsky", "chromaticRate"] * 100
]
plt.boxplot(data, labels=["Bach", "Tchaikovsky"], showfliers=True)
plt.ylabel("Events containing a semitone pair (%)")
plt.title("Chromatic sonority rate: Bach vs. Tchaikovsky")

plt.savefig("bach_tchaikovsky_chromatic_rate.png")
#plt.show()

print("\nPiece counts:")
print(piece_stats.groupby("composer").size())
print("\nMedians:")
print(piece_stats.groupby("composer")[["harmonicVocabulary","harmonicEntropy","chromaticRate","avgSonoritySize"]].median())

#random testing stuff

top_bach = (
    piece_stats[piece_stats["composer"] == "Tchaikovsky"]
    .sort_values("chromaticRate", ascending=False)
    .head(10)
)

print(top_bach[["file", "chromaticRate"]])
