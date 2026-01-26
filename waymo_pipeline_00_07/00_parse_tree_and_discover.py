import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wod_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    args = parser.parse_args()

    tfrecord_files = []
    print(f"[Discovery] Scanning: {args.wod_root}")
    
    for root, dirs, files in os.walk(args.wod_root):
        # DA 및 Unlabeled 제외
        if "domain_adaptation" in root or "unlabeled" in root:
            continue
        for f in files:
            if f.endswith(".tfrecord"):
                tfrecord_files.append(os.path.join(root, f))

    tfrecord_files.sort()
    print(f"[Discovery] Found {len(tfrecord_files)} TFRecord files.")
    
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "discovery_manifest.json"), 'w') as f:
        json.dump(tfrecord_files, f, indent=2)

if __name__ == "__main__":
    main()
