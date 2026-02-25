# RUNNING THE FTC ANALYSIS SCRIPT (STUDENT GUIDE)

This guide explains how to download and run the FTC analysis script on Windows.

---

## Requirements

You will need:

- A Windows PC  
- Internet access  
- Python (3.9+ recommended)  
- Git  

If you are unsure whether these are installed, ask your supervisor.

---

# First-Time Setup

Complete this section **once only**.

---

## Step 1 — Install Git (First Time Only)

1. Download Git:  
   https://git-scm.com/download/win  
2. Install using the default settings.  
3. Restart Command Prompt after installation.

---

## Step 2 — Open Command Prompt

Press:

```
Windows key → type cmd → Enter
```

You should see something like:

```
C:\Users\YourName>
```

---

## Step 3 — Download the Analysis Script

Run:

```bash
git clone https://github.com/JTPiingtoh/FTC-analysis.git
```

This downloads the script to your computer.

---

## Step 4 — Enter the Script Folder

```bash
cd FTC-analysis
```

Your prompt should now look like:

```
C:\Users\YourName\FTC-analysis>
```

---

## Step 5 — Create a Virtual Environment Using venv

```bash
python -m venv .venv
```

Python scripts such as FTC-analysis have dependencies. In other words, the script imports and uses external Python packages that are not stored inside this repository.

To ensure the script runs with the correct dependency versions, we use a virtual environment (venv). A virtual environment creates an isolated Python environment, acting as  a contained “bubble”  dedicated to this project.

This isolation ensures:

- The script uses the exact package versions specified in requirements.txt.

- Other Python projects on your computer do not interfere.

- You avoid accidentally importing incompatible module versions.

- System-wide Python installations remain unchanged.

---

## Step 6 — Activate the Environment

```bash
.venv\Scripts\activate.bat
```

You should now see:

```
(.venv) C:\Users\YourName\FTC-analysis>
```

---

## Step 7 — Install Required Packages

```bash
pip install -r requirements.txt
```

This may take a few minutes.

---

# IMPORTANT — Use a Short File Path

Windows has a file path length limit.

Before running the script, move your image folder to a short path, for example:

```
C:\data\images
```

Avoid:

- OneDrive folders  
- Deep folder structures  
- Very long folder names  

---

## Step 8 — Run the Script

```bash
python main.py
```

A folder picker will appear.  
Select your image folder (e.g., `C:\data\images`).

---

## Step 9 — Outputs

The script will create a new folder:

```
<your folder> OUTPUTS - <timestamp>
```

This folder contains:

- Processed images  
- Results files  

---

# Updating the Script (Important)

If the script has been updated:

1. Open Command Prompt  
2. Navigate to the script folder:

```bash
cd FTC-analysis
```

3. Pull the latest version:

```bash
git pull
```

Then run:

```bash
python main.py
```

---

# Troubleshooting

### Error: “git is not recognised”

Restart Command Prompt after installing Git.

---

### Error: “python is not recognised”

Python is either not installed or not added to PATH.  
Contact your supervisor.

---

### Error: Filename too long (WinError 206)

Move your image folder to a shorter path, such as:

```
C:\data\images
```

---

### Virtual Environment Not Activating

Try:

```bash
.venv\Scripts\activate
```

---

# Running the Analysis Multiple Times

You only need to download the script **once**.

Do **not** run `git clone` again.

---

## ❌ Common Mistake

If you run:

```bash
git clone https://github.com/JTPiingtoh/FTC-analysis.git
```

You may see:

```text
fatal: destination path 'FTC-analysis' already exists and is not an empty directory.
```

This is normal — it means the script is already installed.

---

# How to Run the Analysis Again

## Step 1 — Open Command Prompt

```
Windows key → type cmd → Enter
```

## Step 2 — Go into the Script Folder

```bash
cd FTC-analysis
```

## Step 3 — Activate the Environment

```bash
.venv\Scripts\activate.bat
```

You should see:

```
(.venv) C:\Users\YourName\FTC-analysis>
```

## Step 4 — Run the Script

```bash
python main.py
```

Then select your new image folder.

---

# Simple Rule

- **First time →** use `git clone`  
- **Every time after →**
  ```bash
  cd FTC-analysis
  python main.py
  ```

Never clone the repository twice.