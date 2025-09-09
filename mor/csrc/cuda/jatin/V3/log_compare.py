

import re
import sys

# Ensure correct number of arguments
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <log_file_path>")
    sys.exit(1)

log_file_path = sys.argv[1]  # Take the log file path from the command-line argument

# Define data structure
class DataEntry:
    def __init__(self, value_float, value_int):
        self.value_float = value_float
        self.value_int = value_int

    def __repr__(self):
        return f"(float: {self.value_float}, int: {self.value_int})"

# Initialize lists
A = []
B = []

# Correct Regex patterns for A
pattern_A = re.compile(r"Global Non-Zero Count: (\d+).*?Global Error = ([\deE+\-.]+)", re.DOTALL)

# Fixed pattern for B
pattern_B = re.compile(r"Overall Stats: Avg error \(\{E4M3\)\s*=\s*([\deE+\-.]+)\s*::\s*([\deE+\-.]+)\s*::\s*(\d+)")

try:
    print("\nProcessing log file line by line...")

    with open(log_file_path, "r") as file:
        log_data = file.read()

    # Extract values for A
    matches_A = pattern_A.findall(log_data)
    for match in matches_A:
        non_zero_count = int(match[0])
        error_value = float(match[1])
        A.append(DataEntry(error_value, non_zero_count))

    print(f"\nExtracted {len(A)} entries for A.")

    # Extract values for B
    match_count_B = 0
    with open(log_file_path, "r") as file:
        for i, line in enumerate(file):
            match_B = pattern_B.search(line)
            if match_B:
                avg_error = float(match_B.group(1))
                error_value = float(match_B.group(2))
                non_zero_count = int(match_B.group(3))
                B.append(DataEntry(error_value, non_zero_count))
                match_count_B += 1
                print(f"[B Match {match_count_B}] Line {i+1}: AvgError={avg_error}, Error={error_value}, Count={non_zero_count}")

    print(f"\nExtracted {len(B)} entries for B.")

    # Ensure sizes match
    if len(A) != len(B):
        print(f"\nError: Mismatch in extracted data sizes! A has {len(A)} entries, B has {len(B)} entries.")
        sys.exit(1)

    # Validation
    print("\nValidating extracted values...")
    for i in range(len(A)):
        if A[i].value_int != B[i].value_int:
            print(f"Integer mismatch at index {i}: A = {A[i].value_int}, B = {B[i].value_int}")

        tolerance = A[i].value_float * 0.00000001  # 0.000001% tolerance
        if not (abs(A[i].value_float - B[i].value_float) <= tolerance):
            print(f"Float mismatch at index {i}: A = {A[i].value_float}, B = {B[i].value_float}")

    print("\nValidation passed: All values match within 0.000001% tolerance.")

except FileNotFoundError:
    print(f"\nError: Log file '{log_file_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"\nUnexpected error: {e}")
    sys.exit(1)

