import os
import csv

base_path = "tensor_results"
output_file = "all_results.csv"

step_order = ["1_step", "4_step", "8_step"]  # Para asegurar orden correcto

with open(output_file, "w", newline="") as outfile:
    writer = None  # CSV writer
    for based in ["flop_based", "param_based"]:
        based_path = os.path.join(base_path, based)
        for model in sorted(os.listdir(based_path)):
            model_path = os.path.join(based_path, model)
            for step in step_order:
                step_path = os.path.join(model_path, step)
                filenames = sorted(os.listdir(step_path))  # Orden lexicográfico
                for filename in filenames:
                    if filename.endswith(".csv"):
                        file_path = os.path.join(step_path, filename)

                        # Parse filename
                        parts = filename.replace(".csv", "").split("_")
                        images = parts[2]
                        batch_size = parts[4]

                        # Read avg row
                        with open(file_path) as f:
                            reader = csv.reader(f)
                            headers = next(reader)
                            avg_row = None
                            for row in reader:
                                if row[0].strip().lower() == "avg":
                                    avg_row = row
                                    break

                            if avg_row is not None:
                                # Write header only once
                                if writer is None:
                                    writer = csv.writer(outfile)
                                    new_header = ["based", "model", "step", "images", "batch_size"] + headers[1:]
                                    writer.writerow(new_header)

                                writer.writerow([based, model, step, images, batch_size] + avg_row[1:])
