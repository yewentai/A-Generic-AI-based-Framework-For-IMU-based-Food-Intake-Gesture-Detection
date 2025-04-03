import os


def find_and_combine_files(root_dir, output_file):
    extensions = ".py"
    with open(output_file, "w", encoding="utf-8") as outfile:
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(extensions):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(f"----- Start of {file} -----\n")
                        outfile.write(infile.read())
                        outfile.write(f"\n----- End of {file} -----\n\n")


# Replace 'your_project_directory' with the path to your project directory.
# Replace 'combined_output.txt' with your desired output file name.
find_and_combine_files("./", "combined_output.txt")
