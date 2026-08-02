# https://www.biz-genius.com/tools/publication-trend/

import matplotlib.pyplot as plt

# Data
years = [
    2016, 2017, 2018, 2019, 2020,
    2021, 2022, 2023, 2024, 2025,
    2026
]

counts = [
    0, 0, 0, 1, 0,
    6, 7, 16, 12, 31,
    38
]

# Plot
plt.figure(figsize=(10, 4))
plt.bar(years, counts)
plt.xlabel("Year")
plt.ylabel("Number of Publications")
plt.title("Publication Trend: Quantum Adversarial Machine Learning")
plt.tight_layout()

plt.show()
