import csv
import sys

def filter_csv(input_path, output_path):
    with open(input_path, newline="") as infile, open(output_path, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=["class", "x", "y", "z"])
        writer.writeheader()
        for row in reader:
            writer.writerow({"class": row["class"], "x": row["x"], "y": row["y"], "z": row["z"]})

if __name__ == "__main__":
    if len(sys.argv) != 3:
        input_file = "../../runs/current/detections.csv"
        output_file = "../../runs/current/detections_out.csv"
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    filter_csv(input_file, output_file)