## The North Face E-commerce Intelligence: Product Grouping and Recommender System

### 🚀 Project Overview

This project focuses on leveraging **Machine Learning (ML)** to enhance the online shopping experience and boost conversion rates for **The North Face**'s e-commerce platform.

The marketing department has identified two key areas where ML can have a significant impact:

1.  **Personalized Recommendations:** Deploying a simple recommender system to suggest products similar to those a user is currently viewing, primarily through a "you might also be interested by these products..." section on product pages.
2.  **Catalog Optimization:** Improving the product catalog structure by using unsupervised methods (topic extraction) to challenge and potentially refine existing product categories for better website navigation.

-----

### 🎯 Goals

The project is structured into three main steps:

1.  **Product Grouping:** Identify and group products with similar descriptions using clustering techniques.
2.  **Simple Recommender System:** Utilize the product groups (clusters) to build a basic recommender system.
3.  **Topic Modeling:** Automatically assess the latent topics present in the product descriptions to explore new categorization structures.

-----

### 🖼️ Scope

We will be working with a corpus of **item descriptions** from The North Face's product catalog.

**Data Source:** [👉👉 The North Face product catalog 👈👈](https://www.google.com/search?q=link-to-be-inserted-by-implementer)

**Notebok:** a notebook which contains all the code for the project

**requirements.txt file:** to install all dependencies 

-----

### 🛠️ Technical Implementation

#### Preprocessing of Textual Data

The initial step involves preparing the text data for ML algorithms.

  * **Tools:** `pandas`, `spacy`, `scikit-learn`
  * **Steps:**
      * Clean the corpus: Handle **stop words** and perform **lemmatization**.
      * Encode the cleaned texts using **TF-IDF transformation** (`scikit-learn`'s `TfidfVectorizer`).

#### Part 1: Groups of Products with Similar Descriptions (Clustering)

This phase aims to create distinct groups of functionally or descriptively similar products.

  * **Model:** **DBSCAN**
  * **Metric:** **Cosine distance** (suitable for textual data/TF-IDF matrix).
  * **Parameters:** Tune `eps` and `min_samples` to achieve:
      * $10-20$ clusters.
      * Minimize the number of outliers (products not assigned to a cluster).
  * **Analysis:** Visualize the results by generating a **word cloud** for each cluster to interpret the product groups.

#### Part 2: Simple Recommender System

The identified clusters from Part 1 will form the basis of a basic collaborative filtering approach: products within the same cluster are considered similar.

  * **Function:** Implement a function named `find_similar_items(item_id)`:
      * **Input:** An `item_id` (ID of the product the user is interested in).
      * **Output:** A list of **5 item IDs** belonging to the *same cluster* as the input product.

#### Part 3: Topic Modeling

This part is independent and focuses on uncovering underlying themes (topics) in the product descriptions, offering an alternative view on categorization.

  * **Model:** **Latent Semantic Analysis (LSA)** using **`TruncatedSVD`**
  * **Input:** The TF-IDF matrix.
  * **Parameters:** Try values for `n_components` to extract $10-20$ topics.
  * **Output:** Save the encoded matrix as `topic_encoded_df`.
  * **Interpretation:** Since LSA maps a document to a *mixture* of topics, simplify interpretation by determining the **main topic** for each document.
  * **Analysis:** Generate **word clouds** for each topic to aid interpretation.


# 💻 Installation
## Prerequisites

    Python 3.11 or higher
    Conda (recommended) or pip

## Steps

    Clone the repository

    Create a virtual environment

    Install dependencies

pip install -r requirements.txt

    Launch Jupyter Notebook

jupyter notebook

    Open notebook.ipynb and run the cells

---

## 👤 Author

**Romano Albert**
- 🔗 [LinkedIn](www.linkedin.com/in/albert-romano-ter0rra)
- 🐙 [GitHub](https://github.com/Ter0rra)


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Jedha** for the online training
- **Walmart** for the business case and data
- **Scikit-learn** community for excellent ML tools
- **Pandas** and **NumPy** for data manipulation capabilities

---

## 📞 Support

Questions or suggestions?
- Open an issue on GitHub
- Contact via email
- Connect on LinkedIn

---



