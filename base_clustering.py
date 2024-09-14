import pandas as pd

# Define file paths
file1 = 'path_to_your_first_file.csv'
file2 = 'path_to_your_second_file.csv'

# Define the mapping for one-hot encoding
one_hot_mapping = {
    'CD20': 1,
    'CD66b': 2,
    'CD8, CD20': 3,
    'CTLA4': 4,
    'CD4': 5,
    'CD4, CTLA4': 6,
    'CD8': 7,
    'CD56': 8,
    'CD4, CD8': 9,
    'CD56, CD4': 10,
    'CTLA4, CD8': 11,
    'CD4, CTLA4, CD8': 12,
    'CD4, CD20': 13,
    'CD56, CD4, CTLA4, CD8': 14
}

# Function to apply the one-hot encoding based on the mapping
def one_hot_encode(value):
    if pd.isna(value):  # Handling NaN values
        return 0
    return one_hot_mapping.get(value, 0)

# Load the first file
df1 = pd.read_csv(file1)
# Apply the encoding to the 'expressed_markers' column
df1['Cluster'] = df1['expressed_markers'].apply(one_hot_encode)
# Save the file back
df1.to_csv(file1, index=False)

# Load the second file
df2 = pd.read_csv(file2)
# Apply the encoding to the 'expressed_markers' column
df2['Cluster'] = df2['expressed_markers'].apply(one_hot_encode)
# Save the file back
df2.to_csv(file2, index=False)

print("Encoding completed and files saved.")
