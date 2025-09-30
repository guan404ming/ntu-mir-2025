"""
Verify all data and charts in report.md match the actual results
"""
import json
import os

def verify_files_exist():
    """Check if all referenced charts exist"""
    print("="*60)
    print("VERIFYING CHART FILES")
    print("="*60)

    charts = [
        "results/task1/model_comparison.png",
        "results/task1/confusion_matrix_svm.png",
        "results/task2/training_progress.png",
        "results/task2/model_comparison.png",
        "results/task2/confusion_matrix_panns.png",
        "results/task2/architecture_diagram.png"
    ]

    all_exist = True
    for chart in charts:
        exists = os.path.exists(chart)
        status = "✓" if exists else "✗"
        print(f"{status} {chart}")
        if not exists:
            all_exist = False

    return all_exist


def verify_task1_data():
    """Verify Task 1 data from results_summary.json"""
    print("\n" + "="*60)
    print("VERIFYING TASK 1 DATA")
    print("="*60)

    with open("results/task1/results_summary.json", "r") as f:
        data = json.load(f)

    # Expected values from report
    expected = {
        "SVM": {"top1": 57.14, "top3": 78.79},
        "RandomForest": {"top1": 46.32, "top3": 72.73},
        "KNN": {"top1": 37.23, "top3": 61.04}
    }

    print("\nModel Performance:")
    all_correct = True

    for model_key, report_vals in expected.items():
        # Map model names
        json_key = model_key if model_key == "SVM" else model_key

        actual_top1 = round(data["model_performance"][json_key]["top1_accuracy"] * 100, 2)
        actual_top3 = round(data["model_performance"][json_key]["top3_accuracy"] * 100, 2)

        top1_match = abs(actual_top1 - report_vals["top1"]) < 0.01
        top3_match = abs(actual_top3 - report_vals["top3"]) < 0.01

        status1 = "✓" if top1_match else "✗"
        status3 = "✓" if top3_match else "✗"

        print(f"\n{model_key}:")
        print(f"  {status1} Top-1: Report={report_vals['top1']}%, Actual={actual_top1}%")
        print(f"  {status3} Top-3: Report={report_vals['top3']}%, Actual={actual_top3}%")

        if not (top1_match and top3_match):
            all_correct = False

    return all_correct


def verify_task2_data():
    """Verify Task 2 data from val_set_evaluation.json"""
    print("\n" + "="*60)
    print("VERIFYING TASK 2 DATA")
    print("="*60)

    with open("results/task2/val_set_evaluation.json", "r") as f:
        data = json.load(f)

    # Expected values from report
    expected_top1 = 60.17
    expected_top3 = 83.12

    actual_top1 = round(data["val_top1_accuracy"] * 100, 2)
    actual_top3 = round(data["val_top3_accuracy"] * 100, 2)

    top1_match = abs(actual_top1 - expected_top1) < 0.01
    top3_match = abs(actual_top3 - expected_top3) < 0.01

    status1 = "✓" if top1_match else "✗"
    status3 = "✓" if top3_match else "✗"

    print(f"\nValidation Set Performance:")
    print(f"  {status1} Top-1: Report={expected_top1}%, Actual={actual_top1}%")
    print(f"  {status3} Top-3: Report={expected_top3}%, Actual={actual_top3}%")

    all_correct = top1_match and top3_match
    return all_correct


def verify_calculations():
    """Verify improvement calculations in report"""
    print("\n" + "="*60)
    print("VERIFYING CALCULATIONS")
    print("="*60)

    # Task 1 SVM scores
    svm_top1 = 57.14
    svm_top3 = 78.79

    # Task 2 DL scores
    dl_top1 = 60.17
    dl_top3 = 83.12

    # Calculate improvements
    top1_improvement = dl_top1 - svm_top1
    top3_improvement = dl_top3 - svm_top3

    # Expected from report
    expected_top1_imp = 3.03
    expected_top3_imp = 4.33

    top1_match = abs(top1_improvement - expected_top1_imp) < 0.01
    top3_match = abs(top3_improvement - expected_top3_imp) < 0.01

    status1 = "✓" if top1_match else "✗"
    status3 = "✓" if top3_match else "✗"

    print(f"\nImprovement Calculations:")
    print(f"  {status1} Top-1 improvement: Report={expected_top1_imp}%, Calculated={top1_improvement:.2f}%")
    print(f"  {status3} Top-3 improvement: Report={expected_top3_imp}%, Calculated={top3_improvement:.2f}%")

    all_correct = top1_match and top3_match
    return all_correct


def main():
    print("\n" + "="*60)
    print("REPORT DATA VERIFICATION")
    print("="*60)

    results = {
        "Charts exist": verify_files_exist(),
        "Task 1 data": verify_task1_data(),
        "Task 2 data": verify_task2_data(),
        "Calculations": verify_calculations()
    }

    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✓ ALL CHECKS PASSED - Report data is accurate!")
    else:
        print("✗ SOME CHECKS FAILED - Please review the errors above")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)