import os
from pathlib import Path

# Define lists of names that are typically female or male based on the actual files
FEMALE_NAMES = {
    'abigail', 'alice', 'cora', 'elena', 'emily', 'gianna', 'jade', 'layla',
    'olivia', 'vee', 'saba', 'nyanner', 'nyan'  # Taylor can be either, leaving it out for now
}

MALE_NAMES = {
    'adrian', 'alexander', 'asmon', 'austin', 'axel', 'connor', 'dylan',
    'eli', 'everett', 'explainy', 'gabriel', 'henry', 'ian', 'jeremiah',
    'julian', 'leonardo', 'michael', 'miles', 'nlion', 'paul', 'ryan',
    'thomas', 'wes'
}

# Note: Some names like 'Jordan' can be both, but based on common usage
# I'll put it in female since it's only one file

def rename_voice_files():
    voices_dir = Path("voices")
    
    if not voices_dir.exists():
        print(f"Directory {voices_dir} does not exist!")
        return
    
    files_renamed = 0
    files_skipped = 0
    
    for file_path in voices_dir.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3']:
            original_name = file_path.stem.lower()
            suffix = file_path.suffix
            
            # Check if already has our prefix
            if original_name.startswith('af_') or original_name.startswith('am_'):
                print(f"Skipping {file_path.name} - already has prefix")
                files_skipped += 1
                continue
            
            # Determine if it's female, male, or unknown
            if original_name in FEMALE_NAMES:
                new_name = f"af_{file_path.stem}{suffix}"
                print(f"Female voice detected: {file_path.name} -> {new_name}")
                
            elif original_name in MALE_NAMES:
                new_name = f"am_{file_path.stem}{suffix}"
                print(f"Male voice detected: {file_path.name} -> {new_name}")
                
            else:
                print(f"Uncertain gender: {file_path.name} - left unchanged")
                files_skipped += 1
                continue  # Skip files that don't match known names
            
            # Rename the file
            new_path = file_path.parent / new_name
            file_path.rename(new_path)
            files_renamed += 1
    
    print(f"\nRenaming complete!")
    print(f"Files renamed: {files_renamed}")
    print(f"Files skipped: {files_skipped}")

if __name__ == "__main__":
    rename_voice_files()