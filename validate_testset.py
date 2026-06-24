import os
import sys
from config import cfg


def validate_testset():
    print("=" * 80)
    print("TEST SET VALIDATION")
    print("=" * 80)

    cfg_file = "configs/SDF-Net_Test.yml"
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    dataset_path = cfg.DATASETS.ROOT_DIR
    query_dir = os.path.join(dataset_path, "query")
    gallery_dir = os.path.join(dataset_path, "bounding_box_test")

    results = {
        "path_valid": False,
        "query_count": 0,
        "gallery_count": 0,
        "query_pass": False,
        "gallery_pass": False,
        "query_tif_count": 0,
        "gallery_tif_count": 0,
        "tif_pass": False,
        "total_pass": False
    }

    print(f"\n1. Path Configuration:")
    print(f"   - Dataset root: {dataset_path}")
    print(f"   - Query directory: {query_dir}")
    print(f"   - Gallery directory: {gallery_dir}")

    if os.path.exists(dataset_path) and os.path.exists(query_dir) and os.path.exists(gallery_dir):
        results["path_valid"] = True
        print("   ✓ All directories exist")
    else:
        print("   ✗ Some directories are missing")
        return results

    print(f"\n2. File Format Check (PNG/JPG only):")

    import glob
    query_tif = glob.glob(os.path.join(query_dir, "*.tif"))
    gallery_tif = glob.glob(os.path.join(gallery_dir, "*.tif"))
    
    query_png_jpg = glob.glob(os.path.join(query_dir, "*.png")) + \
                    glob.glob(os.path.join(query_dir, "*.jpg")) + \
                    glob.glob(os.path.join(query_dir, "*.jpeg"))
    
    gallery_png_jpg = glob.glob(os.path.join(gallery_dir, "*.png")) + \
                      glob.glob(os.path.join(gallery_dir, "*.jpg")) + \
                      glob.glob(os.path.join(gallery_dir, "*.jpeg"))

    results["query_tif_count"] = len(query_tif)
    results["gallery_tif_count"] = len(gallery_tif)
    
    print(f"   - Query TIF files: {len(query_tif)}")
    print(f"   - Gallery TIF files: {len(gallery_tif)}")
    print(f"   - Query PNG/JPG files: {len(query_png_jpg)}")
    print(f"   - Gallery PNG/JPG files: {len(gallery_png_jpg)}")

    if len(query_tif) == 0 and len(gallery_tif) == 0:
        results["tif_pass"] = True
        print("   ✓ No TIF files found (only PNG/JPG allowed)")
    else:
        print("   ✗ TIF files found (not allowed)")

    print(f"\n3. Image Count Validation:")
    print(f"   - Required: Query > 9200, Gallery > 9100")

    query_count = len(query_png_jpg)
    gallery_count = len(gallery_png_jpg)
    
    results["query_count"] = query_count
    results["gallery_count"] = gallery_count

    print(f"   - Query images: {query_count}")
    print(f"   - Gallery images: {gallery_count}")

    if query_count > 9200:
        results["query_pass"] = True
        print("   ✓ Query count meets requirement (> 9200)")
    else:
        print("   ✗ Query count does not meet requirement")

    if gallery_count > 9100:
        results["gallery_pass"] = True
        print("   ✓ Gallery count meets requirement (> 9100)")
    else:
        print("   ✗ Gallery count does not meet requirement")

    print(f"\n4. Configuration Settings:")
    print(f"   - TEST.IMS_PER_BATCH: {cfg.TEST.IMS_PER_BATCH}")
    print(f"   - DATALOADER.NUM_WORKERS: {cfg.DATALOADER.NUM_WORKERS}")

    print(f"\n{'=' * 80}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    
    all_pass = all([
        results["path_valid"],
        results["query_pass"],
        results["gallery_pass"],
        results["tif_pass"]
    ])
    results["total_pass"] = all_pass

    print(f"\n{'Item':<30} {'Result':<10} {'Status':<10}")
    print(f"{'-----':<30} {'------':<10} {'------':<10}")
    print(f"{'Path Configuration':<30} {'Valid':<10} {'✓ PASS' if results['path_valid'] else '✗ FAIL'}")
    print(f"{'No TIF files':<30} {'0':<10} {'✓ PASS' if results['tif_pass'] else '✗ FAIL'}")
    print(f"{'Query images (>9200)':<30} {f'{query_count}':<10} {'✓ PASS' if results['query_pass'] else '✗ FAIL'}")
    print(f"{'Gallery images (>9100)':<30} {f'{gallery_count}':<10} {'✓ PASS' if results['gallery_pass'] else '✗ FAIL'}")
    print(f"\n{'Overall':<30} {'':<10} {'✓ ALL PASS' if all_pass else '✗ SOME FAIL'}")

    if all_pass:
        print("\n✓ Test set validation passed! Ready for testing.")
    else:
        print("\n✗ Test set validation failed. Please check the issues above.")

    return results


if __name__ == "__main__":
    validate_testset()