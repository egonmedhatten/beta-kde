import sys
import os
import pytest

def main():
    # Get the absolute path to the 'src' directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, "src")
    
    # Add 'src' to sys.path so 'beta_kernel' can be imported by tests
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"Added {src_path} to sys.path")

    print("Running test suite...")
    print("-" * 50)
    
    # Run pytest on the entire 'tests' directory
    # -v: Verbose
    # -rA: Report all (shows output for passed tests too)
    exit_code = pytest.main(["tests", "-v", "-rA"])
    
    print("-" * 50)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()