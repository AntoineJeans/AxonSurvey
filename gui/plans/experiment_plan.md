# 📄 Comparison Results Page Plan

## **Experiment Info**

* **Experiment ID:**
* **Experiment Date:**

---

## **📊 Comparison Groups**

### **Groups Defined**

not just 2 groups, any number of groups here

#### **Group 1 (Name)**

* **Rats:** 1 rat — *list them*
* **Regions:** 1 region — *list them*
* **Associated ROIs:** *(collapsible)*

  * List of strings (paths) presented as a bullet list inside a collapsible element.

#### **Group 2 (Name)**

* **Rats:** 1 rat — *list them*
* **Regions:** 1 region — *list them*
* **Associated ROIs:** *(collapsible)*

  * List of strings (paths) presented as a bullet list inside a collapsible element.

---

## **🧠 Inference Results**

* **Main Result Image:**
  Image of inference results between the groups.
  Location:

  ```
  ./figures/experiment_id/inference_results/*.png
  ```

* **Per-Group Details** *(collapsible for each group)*:

  * Average density of points for that group
  * Associations between predicted density and individual ROIs in that group
  * `model_name` used for inference
  * Model expected RMSE for individual image patches

---

## **📈 Additional Information**

* **Model Performance Images:**
  Location:

  ```
  ./figures/experiment_id/model_performances/*.png
  ```

  Display all images one after another.

* **UI Feature:**
  All images should have an **"Open Image"** button to open in a new browser tab.

* **Future Expansion:**
  Maybe some other stuff later.


  schema for data inputted into the flask template for rendering 

  comparison_results_data = {
    "experiment_id": None,  # str or int, e.g., "exp_003"
    "experiment_date": None,  # str in YYYY-MM-DD format

    "groups": [
        {
            "name": None,  # str, e.g., "Group A"
            "rats": [],  # list of str, e.g., ["rat_01"]
            "regions": [],  # list of str, e.g., ["region_hippocampus"]

            # Inference-related fields
            "model_name": None,  # str
            "expected_rmse": None,  # float
            "average_density": None,  # float or formatted str
            "roi_associations": {}  # dict mapping ROI path -> predicted density
        },
        {
            "name": None,
            "rats": [],
            "regions": [],
            "model_name": None,
            "expected_rmse": None,
            "average_density": None,
            "roi_associations": {}
        }
    ],

    # Paths will point to files under /static/experiment_graphs/
    "main_inference_image": None,  # str, e.g., "/static/experiment_graphs/inference_results.png"
    "model_performance_images": []  # list of exactly 3 str paths, e.g., ["/static/experiment_graphs/model_perf1.png", ...]
}
