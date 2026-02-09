# https://www.biz-genius.com/tools/publication-trend/

import matplotlib.pyplot as plt

# Data
years = [
    2010, 2011, 2012, 2013, 2014,
    2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
]
counts = [
    0, 0, 0, 0, 0,
    0, 0, 0, 2, 1, 0, 5, 7, 7, 4, 11
]

# Plot
plt.figure(figsize=(10, 4))
plt.bar(years, counts)
plt.xlabel("Year")
plt.ylabel("Number of Publications")
plt.title("Publication Trend: Quantum Adversarial Machine Learning")
plt.tight_layout()

plt.show()
