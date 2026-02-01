import csv
import sys
import os

input_file = ""

if len(sys.argv) < 1:
        print("This script expects a filename.")
        sys.exit(1)
else:
    input_file = sys.argv[1]

input_filepath = os.path.abspath(os.path.join("data", "raw", input_file))

output_file = f"sorted_{input_file}"

output_filepath = os.path.abspath(os.path.join("data", "raw", output_file))

with open(input_filepath, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Preserve header (first row) and sort the remaining rows by the first column
if not data:
    data_sorted = []
else:
    header = data[0]
    body = data[1:]
    if body:
        body_sorted = sorted(body, key=lambda row: row[0])
    else:
        body_sorted = []
    data_sorted = [header] + body_sorted

with open(output_filepath, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_sorted)

print(f"CSV sorted successfully and saved to {output_filepath}")