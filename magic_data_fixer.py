import csv
import sys

input_file = ""

if len(sys.argv) < 1:
        print("This script expects a filename.")
        sys.exit(1)
else:
    input_file = sys.argv[1]

output_file = f"sorted_{input_file}"

with open(input_file, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Sort by the first column (index 0)
data_sorted = sorted(data, key=lambda row: row[0])

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_sorted)

print(f"CSV sorted successfully and saved to {output_file}")