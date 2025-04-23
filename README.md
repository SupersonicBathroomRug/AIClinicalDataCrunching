# AI Clinical Trial Data Analysis

A Python project for analyzing clinical trial data.

## Installation Guide for Beginners

This guide will help you set up and run this project even if you've never coded before.

### Step 1: Check Python Installation

1. First, check if Python is already installed:
   - On Windows: 
     1. Press Windows key + R
     2. Type "cmd" and press Enter
     3. Or: Click Start menu, type "Command Prompt", and click it
     4. Once Command Prompt opens, type `python --version`
   - On Mac/Linux: Open Terminal and type `python3 --version`
2. If you see a version number (like "Python 3.x.x"), Python is already installed - close the command prompt window and skip to Step 2
3. If you get an error, follow these steps to install Python:
   - Go to [Python's official website](https://www.python.org/downloads/)
   - Click the big yellow button that says "Download Python"
   - Once downloaded, open the installer:
     - On Windows: Make sure to check the box that says "Add Python to PATH"
     - Click "Install Now"
   - Wait for the installation to complete and click "Close"
   - Restart your computer for the changes to take effect

### Step 2: Install Visual Studio Code (VS Code)

1. Go to [VS Code's website](https://code.visualstudio.com/)
2. Click the big blue "Download" button
3. Once downloaded, open the installer
4. Follow the installation steps (you can use all default settings)
5. Once installed, open VS Code

### Step 3: Download This Project

1. On this GitHub page, look for a green button that says "Code"
2. Click it and select "Download ZIP"
3. Once downloaded, find the ZIP file in your Downloads folder
4. Right-click the ZIP file and select "Extract All..."
5. Choose where to extract it (e.g., your Documents folder)
6. Click "Extract"

### Step 4: Open the Project in VS Code

1. Open VS Code
2. Go to File → Open Folder
3. Navigate to where you extracted the ZIP file
4. Select the folder and click "Select Folder"
5. If VS Code asks "Do you trust the authors of these files?", click "Yes"

### Step 5: Set Up Python Environment

1. In VS Code, click View → Terminal (or press `` Ctrl + ` ``)
2. In the terminal, type these commands one at a time (press Enter after each):

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 6: Run the Project

1. In VS Code, create a new file called `.env` in the root directory
   - Right-click in the file explorer on the left
   - Select "New File" and name it `.env`
   - Add this line to the file (replace with your actual key):
     ```
     OPENAI_API_KEY=your-api-key-here
     ```
   - Save the file

2. In VS Code, look for the file `main.py` in the file explorer on the left
3. Click on `main.py` to open it
4. Click the "Run" button (triangle/play button) in the top-right corner
   - If VS Code asks you to install Python extensions, click "Install"

### Troubleshooting

If you encounter any issues:

1. **Python not found**:
   - Make sure you checked "Add Python to PATH" during installation
   - Try restarting your computer

2. **Terminal says "cannot be loaded because running scripts is disabled"** (Windows):
   - Open PowerShell as Administrator
   - Run: `Set-ExecutionPolicy RemoteSigned`
   - Type `Y` and press Enter

3. **Permission errors**:
   - Make sure you're running VS Code as an administrator
   - Try creating the project folder somewhere else (e.g., Documents)

4. **Module not found errors**:
   - Make sure you activated the virtual environment (Step 5)
   - Try running the pip install command again

### Need Help?

If you're still having trouble:
1. Make sure you followed each step exactly
2. Try restarting your computer
3. Contact me for help

## For Developers

If you're familiar with Python development, you can simply:

```bash
# Clone the repository
git clone [repository-url]

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

The project runs for the ICTRP-Results.xml file in the root directory, running for the first 10 rows by default. This can be changed in the main.py file.