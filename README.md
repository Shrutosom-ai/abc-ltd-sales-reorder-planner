# ABC Ltd · Sales & Reorder Planner

A decision-support tool for ABC Ltd, a dairy company (name anonymised). For a planned dispatch, it answers two questions:

1. **How much of it will sell?** Linear regression on quantity sold.
2. **Will stock fall below the minimum level, so a reorder is needed?** Logistic regression on a yes/no reorder flag.

**Live app:** `https://<your-app-name>.streamlit.app` (add your link after deploying)

---

## What is in this repository

| File | What it is |
|---|---|
| `ABC_Ltd_Sales_and_Reorder_Model.ipynb` | Colab notebook: data audit, cleaning, exploratory analysis, both regressions, and model export |
| `app.py` | The Streamlit app that managers use |
| `abc_ltd_models.joblib` | The two trained models, created by the notebook |
| `requirements.txt` | Library versions for Streamlit, created by the notebook |

The data file is **not** in the repository. It is 40 MB, over GitHub's 25 MB limit for files uploaded in the browser, and the app does not need it.

## Results in brief

| | |
|---|---|
| Records | 122,983 raw; 92,785 after removing records that break the stock rule (more sold, or more left over, than was available) |
| Train / test | Trained on 2019–2021, tested on 2022 |
| Linear regression (quantity sold) | R² = 0.49 on 2022 data; typical error ±115 L/kg. Only quantity available matters. |
| Logistic regression (reorder needed) | AUC = 0.90 on 2022 data. At the 30% alert level it catches 62% of reorders, and 52% of alerts are right. |
| No measurable effect | Price, product, brand, storage, shelf life, channel, farm and customer state, farm size, land area, herd size, month |

---

## Step by step: Colab → GitHub → Streamlit

### Part 1 · Run the analysis in Google Colab (about 15 minutes)

1. Rename the data file to **`abc_ltd_dairy_data.csv`**. This keeps the real company name out of your notebook outputs.
2. Open [colab.research.google.com](https://colab.research.google.com) and sign in with your Google account.
3. Choose **File → Upload notebook** and pick `ABC_Ltd_Sales_and_Reorder_Model.ipynb`.
4. Choose **Runtime → Run all**.
5. Under *Step 1* a **Choose Files** button appears. Pick `abc_ltd_dairy_data.csv`; the 40 MB upload takes about a minute. Alternatively, drag the file into the **Files** panel (folder icon on the left) before choosing *Run all*.
6. Let every cell finish (about a minute). The printed tables and charts are your results for the report.
7. The last cell downloads **`abc_ltd_model_files.zip`**. Allow the download if the browser asks. Unzip it to get `abc_ltd_models.joblib` and `requirements.txt`.
8. Note the Python version printed in the last cell (for example *Python 3.12*). You need it in Part 3.
9. Save the notebook with its outputs: **File → Download → Download .ipynb**.

> Always upload `abc_ltd_models.joblib` and `requirements.txt` **together, from the same Colab run**. The requirements file pins the scikit-learn and numpy versions that saved the model; a mismatch stops the app from loading.

### Part 2 · Put the files on GitHub (about 10 minutes)

1. Sign in at [github.com](https://github.com), or create a free account.
2. Click **+** (top right) → **New repository**.
3. Name it, for example, `abc-ltd-sales-reorder-planner`. Choose **Public** and leave *Add a README file* unticked. Click **Create repository**.
4. On the new, empty repository page, click **uploading an existing file**.
5. Drag in these five files: `app.py`, `requirements.txt`, `abc_ltd_models.joblib`, `README.md` and your downloaded notebook. **Do not upload the CSV.**
6. Click **Commit changes**.

### Part 3 · Deploy on Streamlit Community Cloud (about 5 minutes)

1. Go to [share.streamlit.io](https://share.streamlit.io) and choose **Continue with GitHub**. Authorise Streamlit when asked.
2. Click **Create app** (top right), then **Yup, I have an app**.
3. Fill in:
   - **Repository:** `your-github-name/abc-ltd-sales-reorder-planner`
   - **Branch:** `main`
   - **File path:** `app.py`
   - **App URL:** a short name, for example `abc-ltd-planner`
4. Click **Advanced settings**, set **Python version** to the one the notebook printed, and click **Save**.
5. Click **Deploy**. The first build takes a few minutes. Your app is then live at `https://<your-app-name>.streamlit.app`.
6. Paste that link at the top of this README.

### Part 4 · Prepare the managerial user study

1. Create your feedback form, for example a Google Form.
2. On GitHub, open `app.py`, click the pencil icon, and paste the form link between the quotes in `FEEDBACK_FORM_URL = ""` near the top. Click **Commit changes**. Streamlit picks up the change within a minute or two.
3. Share the app link with managers and other participants. The app's **Feedback** tab gives them four short tasks to try before they open the form.

---

## Changing things later

- Any file you edit or upload on GitHub is redeployed automatically.
- If you re-run the notebook, upload the new `abc_ltd_models.joblib` **and** `requirements.txt` together.
- The copies of those two files that come with this project were produced by the same notebook (Python 3.11, scikit-learn 1.8.0), so the app also works before you run Colab. Replace them with your own Colab outputs so the app matches your report.

## Troubleshooting

| What you see | What to do |
|---|---|
| *Could not load abc_ltd_models.joblib* | The model file is missing, or `requirements.txt` came from a different run. Upload both files from the same Colab run. |
| *ModuleNotFoundError* | `requirements.txt` is missing or renamed. It must sit next to `app.py` with exactly that name. |
| Build fails while installing numpy or scikit-learn | Streamlit is using a different Python version from Colab. Delete the app in Streamlit and deploy again, choosing the Python version the notebook printed. |
| The Colab upload stalls | Put the CSV in the top level of your Google Drive (*My Drive*), mount Drive with the Drive button in Colab's Files panel, and run the notebook again. It finds the file there. |
| The app shows a *sleeping* screen | Community Cloud pauses apps that have had no visitors for a while. Click the wake-up button and wait about 30 seconds. Open the app yourself shortly before a session with participants. |

## Run the app on your own computer (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data and limitations

- **Source:** a public dairy sales and inventory dataset (full citation in the project report). The company and brand names are anonymised.
- **Data quality:** the audit found values a real business would not produce: expiry dates before production dates (22% of records), more sold or left over than was available (removed), and the same ₹10–₹100 price range for every product. The results demonstrate the method rather than real customer behaviour.
- **Business rule:** if no more than the minimum stock level is made available, the app shows a reorder as certain. The logistic curve cannot show that jump by itself.
