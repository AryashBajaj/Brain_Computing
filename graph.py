import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

a1 = ["1e-5, 1e-4, 1e-3, 1e-2, 1e-1"]
a2 = [
    [57.78, 45, 62, 87],
    [63, 59, 71, 94.8]
]

df = pd.DataFrame(a2, columns=a1)
print(df.head()) 

sns.boxplot(df)
plt.show()
