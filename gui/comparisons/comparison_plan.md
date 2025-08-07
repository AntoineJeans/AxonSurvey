# 🧪 Rat Comparison System – Group-Based ROI Analysis

This page allows the user to define **comparison groups** before running a group-based analysis on rat brain ROIs (regions of interest). Each group is composed of selected rats and selected regions. Once groups are defined, a comparison function will generate summary statistics and visualizations.

---

## 🔧 Group Creation Logic

* Each group is made of:

  * A **subset of rats**
  * A **subset of region types**
* **All slices** for the selected rats will be automatically included (this is hardcoded for now).
* Groups are constructed **before** running any comparison.

---

## 🗂️ UI Structure

* The UI allows for creating **multiple group tabs** (`Group 1`, `Group 2`, etc.).
* Outside the tabs:

  * A **"Create New Group"** button is always visible and can be clicked to add more tabs.
* Inside each group tab:

  * A **"Delete Group"** button allows removing the group.
  * The user can freely switch between tabs in any order.

---

## 🧩 Group Tab Contents

Each group tab is split into two main sections:

1. **Rats Selector**

   * A checklist of all available rats
   * Includes an **"All rats"** option
   * Users can either:

     * Select all rats at once
     * Or check individual rats carefully (encourage caution here)

2. **Regions Selector**

   * A checklist of all available brain regions
   * Includes an **"All regions"** option
   * Users can either:

     * Select all regions at once
     * Or pick specific regions one by one

Once a group is configured, the user can either:

* Proceed to create another group
* Or stop and move to comparison

---

## 📊 Comparison Execution

* After defining the desired number of groups, the user clicks a **"Generate Comparison"** button.
* The system will compute and return:

  * Group-level summary statistics
  * Graphs/images comparing groups
  * Text or numeric results

---

## 📁 Output Area

Reserve space on the page to display:

* Generated graphs (e.g., bar plots, line charts)
* Numerical/textual output from the comparison process
