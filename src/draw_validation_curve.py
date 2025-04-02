#%%
 
import pandas as pd
import matplotlib.pyplot as plt
import glob

files = glob.glob('output/*curve.csv')
plt.figure(figsize=(10, 5))
for file in files:
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['val_loss'], label=file.split('/')[-1].split('.')[0])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.yscale('log')
    plt.grid()
    plt.title('Validation Loss vs Epoch')
    plt.legend()
plt.savefig('output/validation_curve.png')
plt.show()

# %%
