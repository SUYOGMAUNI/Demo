"""
Log Viewer Utility for AI-Based Exam Proctoring System
View and analyze text logs from D:\logs
File: log_viewer.py
"""

import os
from datetime import datetime
from pathlib import Path

LOGS_DIR = r"D:\logs"


class LogViewer:
    """Utility to view and analyze proctoring logs"""

    def __init__(self):
        self.logs_dir = Path(LOGS_DIR)

        if not self.logs_dir.exists():
            print(f"Logs directory not found: {LOGS_DIR}")
            print("No logs to display.")
            return

    def list_all_logs(self):
        """List all available log files"""
        log_files = sorted(self.logs_dir.glob('*.txt'), key=os.path.getmtime, reverse=True)

        if not log_files:
            print("No log files found.")
            return []

        print("\n" + "=" * 70)
        print("AVAILABLE LOG FILES")
        print("=" * 70)

        for idx, log_file in enumerate(log_files, 1):
            file_size = os.path.getsize(log_file)
            modified_time = datetime.fromtimestamp(os.path.getmtime(log_file))

            print(f"{idx}. {log_file.name}")
            print(f"   Size: {file_size} bytes | Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("=" * 70 + "\n")

        return log_files

    def view_log(self, log_file):
        """Display contents of a log file"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            print("\n" + "=" * 70)
            print(f"LOG FILE: {log_file.name}")
            print("=" * 70 + "\n")
            print(content)
            print("\n" + "=" * 70 + "\n")

        except Exception as e:
            print(f"Error reading log file: {e}")

    def search_logs(self, keyword):
        """Search for keyword in all logs"""
        log_files = list(self.logs_dir.glob('*.txt'))

        if not log_files:
            print("No log files found.")
            return

        print(f"\nSearching for: '{keyword}'")
        print("=" * 70)

        found_count = 0

        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                matches = [(i + 1, line.strip()) for i, line in enumerate(lines)
                           if keyword.lower() in line.lower()]

                if matches:
                    found_count += len(matches)
                    print(f"\n{log_file.name}:")
                    for line_num, line in matches:
                        print(f"  Line {line_num}: {line}")

            except Exception as e:
                print(f"Error reading {log_file.name}: {e}")

        print("\n" + "=" * 70)
        print(f"Found {found_count} matches in {len(log_files)} files")
        print("=" * 70 + "\n")

    def analyze_logs_by_candidate(self, candidate_id):
        """Analyze all logs for a specific candidate"""
        log_files = list(self.logs_dir.glob(f'*{candidate_id}*.txt'))

        if not log_files:
            print(f"No logs found for candidate: {candidate_id}")
            return

        print("\n" + "=" * 70)
        print(f"LOGS FOR CANDIDATE: {candidate_id}")
        print("=" * 70 + "\n")

        total_warnings = {
            'audio': 0,
            'face': 0,
            'gaze': 0
        }

        for log_file in sorted(log_files):
            print(f"\n{log_file.name}")
            print("-" * 70)

            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Count warnings
                warning_lines = [line for line in content.split('\n') if 'WARNING' in line]

                if 'audio_' in log_file.name:
                    total_warnings['audio'] += len(warning_lines)
                    print(f"  Audio Warnings: {len(warning_lines)}")
                elif 'facematch_' in log_file.name:
                    total_warnings['face'] += len(warning_lines)
                    print(f"  Face Match Warnings: {len(warning_lines)}")
                elif 'gaze_' in log_file.name:
                    total_warnings['gaze'] += len(warning_lines)
                    print(f"  Gaze Warnings: {len(warning_lines)}")

                # Extract timestamps
                lines = content.split('\n')
                start_time = None
                end_time = None

                for line in lines:
                    if 'Started at:' in line:
                        start_time = line.split('Started at:')[1].strip()
                    elif 'Ended at:' in line:
                        end_time = line.split('Ended at:')[1].strip()

                if start_time:
                    print(f"  Started: {start_time}")
                if end_time:
                    print(f"  Ended: {end_time}")

            except Exception as e:
                print(f"  Error: {e}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total Audio Warnings: {total_warnings['audio']}")
        print(f"Total Face Match Warnings: {total_warnings['face']}")
        print(f"Total Gaze Warnings: {total_warnings['gaze']}")
        print(f"TOTAL WARNINGS: {sum(total_warnings.values())}")
        print("=" * 70 + "\n")

        # Verdict
        total = sum(total_warnings.values())
        if total == 0:
            print("✓ VERDICT: PASS - No violations detected")
        elif total <= 3:
            print("⚠ VERDICT: REVIEW - Minor violations detected")
        else:
            print("✗ VERDICT: FAIL - Major violations detected")
        print()

    def get_recent_violations(self, n=10):
        """Get N most recent violations from all logs"""
        log_files = list(self.logs_dir.glob('*.txt'))

        if not log_files:
            print("No log files found.")
            return

        violations = []

        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    if 'WARNING' in line or 'VIOLATION' in line or 'MISMATCH' in line:
                        violations.append({
                            'file': log_file.name,
                            'line': line.strip(),
                            'time': os.path.getmtime(log_file)
                        })

            except Exception as e:
                continue

        # Sort by time
        violations.sort(key=lambda x: x['time'], reverse=True)

        print("\n" + "=" * 70)
        print(f"RECENT VIOLATIONS (Last {n})")
        print("=" * 70 + "\n")

        for i, v in enumerate(violations[:n], 1):
            print(f"{i}. [{v['file']}]")
            print(f"   {v['line']}\n")

        print("=" * 70 + "\n")

    def delete_old_logs(self, days=7):
        """Delete logs older than N days"""
        from datetime import timedelta

        log_files = list(self.logs_dir.glob('*.txt'))
        cutoff_time = datetime.now() - timedelta(days=days)

        deleted = 0
        for log_file in log_files:
            file_time = datetime.fromtimestamp(os.path.getmtime(log_file))

            if file_time < cutoff_time:
                try:
                    os.remove(log_file)
                    print(f"Deleted: {log_file.name}")
                    deleted += 1
                except Exception as e:
                    print(f"Error deleting {log_file.name}: {e}")

        print(f"\nDeleted {deleted} old log files.")


def main():
    """Interactive log viewer"""
    viewer = LogViewer()

    while True:
        print("\n" + "=" * 70)
        print("LOG VIEWER - AI-BASED EXAM PROCTORING SYSTEM")
        print("=" * 70)
        print("\nOptions:")
        print("  1. List all logs")
        print("  2. View specific log")
        print("  3. Search logs")
        print("  4. Analyze by candidate ID")
        print("  5. View recent violations")
        print("  6. Delete old logs")
        print("  7. Exit")

        choice = input("\nEnter choice (1-7): ").strip()

        if choice == '1':
            viewer.list_all_logs()

        elif choice == '2':
            log_files = viewer.list_all_logs()
            if log_files:
                try:
                    idx = int(input("Enter log number to view: ")) - 1
                    if 0 <= idx < len(log_files):
                        viewer.view_log(log_files[idx])
                    else:
                        print("Invalid log number.")
                except ValueError:
                    print("Invalid input.")

        elif choice == '3':
            keyword = input("Enter search keyword: ").strip()
            if keyword:
                viewer.search_logs(keyword)

        elif choice == '4':
            candidate_id = input("Enter candidate ID: ").strip()
            if candidate_id:
                viewer.analyze_logs_by_candidate(candidate_id)

        elif choice == '5':
            try:
                n = int(input("Number of violations to show (default 10): ").strip() or "10")
                viewer.get_recent_violations(n)
            except ValueError:
                viewer.get_recent_violations(10)

        elif choice == '6':
            try:
                days = int(input("Delete logs older than how many days? (default 7): ").strip() or "7")
                confirm = input(f"Delete logs older than {days} days? (y/n): ").strip().lower()
                if confirm == 'y':
                    viewer.delete_old_logs(days)
            except ValueError:
                print("Invalid input.")

        elif choice == '7':
            print("Exiting...")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()