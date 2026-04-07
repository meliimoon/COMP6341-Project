import os
from fast_plate_ocr import LicensePlateRecognizer
from Levenshtein import distance as levenshtein_distance
import argparse
import json
import re

# Function to calculate character accuracy based on Levenshtein distance
def char_accuracy(pred, gt):
    lev_dist = levenshtein_distance(pred, gt)
    acc = 1 - lev_dist / max(len(gt), 1)
    return acc, lev_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        dest = "input_dir",
        help="the file path of input images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        dest = "output_dir",
        default=os.getcwd(),
        help="the file path for saving fast-plate-OCR results",
    )
    parser.add_argument(
        "-gt",
        "--ground_truth_file",
        type=str,
        dest = "ground_truth_file",
        default=None,
        help="the file path of ground truth images txt file (if not specified, will not calculate accuracy)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)

    # Parsing the GT file into a dictionary
    if args.ground_truth_file is not None:
        gt_dict = {}
        with open(args.ground_truth_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Split on tab
                path, plate = line.split('\t')
                path = path.strip('"')
                filename = os.path.basename(path)
                gt_dict[filename] = plate

    # Initialize fast-plate-OCR model
    model = LicensePlateRecognizer('cct-s-v2-global-model')
    
    input_imgs = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith((".jpg", ".png"))]

    results = {}
    total_lev_dist = 0
    num_gt_samples = 0

    for file in input_imgs:
        pred = model.run(file)[0].plate
        filename = os.path.basename(file)
        filename = re.sub(r'^.*?xLR_', '', filename)
        
        print(f"Detected plate for {file}: {pred}")

        results[filename] = {
            "prediction": pred
        }

        if args.ground_truth_file is not None:
            # Compare the detected plate with the ground truth plate and calculate character accuracy
            if filename in gt_dict:
                gt_plate = gt_dict[filename]
                accuracy, lev_dist = char_accuracy(pred, gt_plate)
                print(f"Ground truth plate: {gt_plate}, Levenshtein distance: {lev_dist}, Character accuracy: {accuracy:.2%}")

                # Store results for JSON output later
                results[filename].update({
                    "ground_truth": gt_plate,
                    "levenshtein_distance": lev_dist,
                    "character_accuracy": accuracy
                })

                # Accumulate stats to be used for average Levenshtein distance calculation
                total_lev_dist += lev_dist
                num_gt_samples += 1

            else:
                print(f"No ground truth plate found for {filename}. Skipping accuracy calculation.")
    
    if num_gt_samples > 0:
        avg_lev_dist = total_lev_dist / num_gt_samples
        print(f"\nAverage Levenshtein distance: {avg_lev_dist:.4f}")
    else:
        avg_lev_dist = None

    # Save results to JSON file
    output_json_path = os.path.join(args.output_dir, "ocr_results.json")

    with open(output_json_path, "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "num_samples": num_gt_samples,
                "average_levenshtein_distance": avg_lev_dist
            }
        }, f, indent=4)
    