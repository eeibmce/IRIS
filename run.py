# run.py
# This is the file you run from the command line to analyze an interview.
#
# Usage examples:
#   python run.py --video interview.mp4
#   python run.py --video interview.mp4 --output report.txt
#   python run.py --transcript "I worked at Acme for 3 years..."
#
# Run "python run.py --help" to see all options.

import sys    # for reading command line arguments and exiting
import os     # for checking if files exist

import pipeline             # the main analysis pipeline
import save_results_table   # writes results into the Excel spreadsheet


def print_help():
    # Prints usage instructions to the console.

    print("")
    print("Interview Authenticity Analyzer")
    print("================================")
    print("")
    print("Usage:")
    print("  python run.py --video path/to/video.mp4")
    print("  python run.py --video path/to/video.mp4 --output report.txt")
    print("  python run.py --transcript \"your text here\"")
    print("  python run.py --audio path/to/audio.wav")
    print("")
    print("Options:")
    print("  --video      path to interview video file (.mp4, .mov, .avi, etc.)")
    print("  --audio      path to separate audio file (optional)")
    print("  --transcript text to analyze instead of recording")
    print("  --output     save report to this file (optional)")
    print("  --help       show this help message")
    print("")


def parse_arguments():
    # Reads the command line arguments and returns them as a dictionary.
    # For example: python run.py --video test.mp4 --output report.txt
    # Returns: {"video": "test.mp4", "output": "report.txt"}

    args = {
        "video":      None,    # path to video file
        "audio":      None,    # path to audio file
        "transcript": None,    # text to analyze
        "output":     None,    # path to save report
        "xlsx":       "Results_Table.xlsx",  # spreadsheet to update
    }

    # sys.argv is a list of everything typed on the command line
    # sys.argv[0] is always the script name (run.py)
    # sys.argv[1], [2], etc. are the arguments we passed
    all_args = sys.argv[1:]   # everything after "run.py"

    # Loop through the arguments two at a time (flag, value)
    i = 0
    while i < len(all_args):
        arg = all_args[i]   # current argument (e.g. "--video")

        if arg == "--help" or arg == "-h":
            print_help()
            sys.exit(0)   # exit with code 0 (success)

        elif arg == "--video" or arg == "-v":
            # Next argument should be the video path
            if i + 1 < len(all_args):
                args["video"] = all_args[i + 1]
                i += 2   # skip past both --video and the path
            else:
                print("Error: --video requires a file path")
                sys.exit(1)

        elif arg == "--audio" or arg == "-a":
            if i + 1 < len(all_args):
                args["audio"] = all_args[i + 1]
                i += 2
            else:
                print("Error: --audio requires a file path")
                sys.exit(1)

        elif arg == "--transcript" or arg == "-t":
            if i + 1 < len(all_args):
                args["transcript"] = all_args[i + 1]
                i += 2
            else:
                print("Error: --transcript requires text in quotes")
                sys.exit(1)

        elif arg == "--output" or arg == "-o":
            if i + 1 < len(all_args):
                args["output"] = all_args[i + 1]
                i += 2
            else:
                print("Error: --output requires a file path")
                sys.exit(1)

        elif arg == "--xlsx" or arg == "-x":
            if i + 1 < len(all_args):
                args["xlsx"] = all_args[i + 1]
                i += 2
            else:
                print("Error: --xlsx requires a file path")
                sys.exit(1)

        else:
            print("Unknown argument: " + arg)
            print("Run 'python run.py --help' for usage")
            sys.exit(1)

    return args


def check_file_exists(path, label):
    # Checks that a file actually exists before trying to open it.
    # Prints an error and exits if the file is not found.

    if not os.path.exists(path):
        print("Error: " + label + " file not found: " + path)
        sys.exit(1)


def save_report(result, output_path):
    # Saves a text report of the analysis result to a file.

    # Build the report as a list of text lines
    lines = []
    lines.append("INTERVIEW AUTHENTICITY ANALYSIS REPORT")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Decision:    " + result["decision"].upper())
    lines.append("Confidence:  " + str(round(result["confidence"] * 100, 1)) + "%")
    lines.append("Uncertainty: " + str(round(result["uncertainty"] * 100, 1)) + "%")
    lines.append("Loop iters:  " + str(result.get("closed_loop_iterations", 0)))
    lines.append("")
    lines.append("Evidence:")

    # Sort by weight, highest first
    sorted_evidence = sorted(result["evidence"], key=lambda e: e["weight"], reverse=True)

    for item in sorted_evidence:
        lines.append("  [" + item["type"] + "] " + item["description"] +
                     " (weight=" + str(round(item["weight"], 2)) + ")")

    lines.append("")
    transcript = result.get("transcript", "")
    if transcript:
        lines.append("Transcript:")
        lines.append(transcript)

    # Join all lines with newlines and write to file
    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Report saved to: " + output_path)


def main():
    # The main function - reads arguments, runs analysis, shows results.

    print("Interview Authenticity Analyzer")
    print("================================")

    # Read command line arguments
    args = parse_arguments()

    # Make sure the user provided at least one input
    if args["video"] is None and args["transcript"] is None and args["audio"] is None:
        print("Error: you must provide --video, --audio, or --transcript")
        print("Run 'python run.py --help' for usage")
        sys.exit(1)

    # Check that any file paths actually exist
    if args["video"] is not None:
        check_file_exists(args["video"], "video")

    if args["audio"] is not None:
        check_file_exists(args["audio"], "audio")

    # Run the analysis pipeline
    result = pipeline.analyze(
        video_path  = args["video"],
        audio_path  = args["audio"],
        transcript  = args["transcript"],
    )

    # Print the result to the console
    pipeline.print_result(result)

    # Save to text report if requested
    if args["output"] is not None:
        save_report(result, args["output"])

    # Update the Results Table spreadsheet
    if args["xlsx"] is not None and os.path.exists(args["xlsx"]):
        save_results_table.save_results_table(result, xlsx_path=args["xlsx"])
    elif args["xlsx"] is not None:
        print("Note: spreadsheet not found at '" + args["xlsx"] + "' - skipping table update")


# This line means: only run main() if this file is run directly.
# If another file imports this file, main() will NOT run automatically.
if __name__ == "__main__":
    main()
