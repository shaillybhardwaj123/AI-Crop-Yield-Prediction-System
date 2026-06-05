import shutil
import os

src = r"c:\Users\hp\Downloads\yield_df.csv"
dst = r"c:\Users\hp\OneDrive\Desktop\yield project\data\yield_df.csv"

try:
    print(f"Attempting to copy from {src} to {dst}")
    shutil.copy(src, dst)
    print("Copy successful!")
except Exception as e:
    print(f"Error copying file: {e}")
