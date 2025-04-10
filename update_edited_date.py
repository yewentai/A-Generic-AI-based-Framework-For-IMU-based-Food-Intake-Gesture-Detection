import subprocess
from datetime import date


def get_changed_python_files():
    result = subprocess.run(["git", "diff", "--name-only"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    files = result.stdout.strip().split("\n")
    return [f for f in files if f.endswith(".py")]


def update_header_date(file_path):
    today = date.today().strftime("%Y-%m-%d")
    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for line in lines:
            if line.strip().startswith("Edited"):
                line = f"Edited      : {today}\n"
            f.write(line)


if __name__ == "__main__":
    py_files = get_changed_python_files()
    for file in py_files:
        try:
            update_header_date(file)
            print(f"Updated: {file}")
        except Exception as e:
            print(f"Error updating {file}: {e}")
