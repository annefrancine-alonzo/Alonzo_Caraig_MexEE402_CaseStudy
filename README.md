# MexEE 402: Data Preprocessing Case Study

**MexEE Elective 2: Data Science and Machine Learning**

Batangas State University, Alangilan Campus


1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Alonzo, Anne Francine B. | 23-01520 | MEXE-4102 |
| Caraig, Charles Edward R.| 23-01535 | MEXE-4102 |

## Notebook links

| Chapter | Link |
|---|---|
| Ch1_2_3 | [https://colab.research.google.com/drive/1qvJiglOfSEUvoAXfr96EJS3Boy58KrFQ#scrollTo=eumOVSMaTv4J] |
| Ch4 | [https://colab.research.google.com/drive/1fKw3U4o5cgcGAEFuR9tfLLJXOBehJ-WB] |
| Ch5 | [https://colab.research.google.com/drive/1Ts7dt_f24ue8CnJk-HOpyFIfSxb3A07e#scrollTo=LGQenL21h7bX]  |
| Ch6 | [https://colab.research.google.com/drive/1ssOABz0sWxhzPNBjGuZZGtHm04zMNUq2]  |
| Ch7 | [https://colab.research.google.com/drive/1PPu8GN0pY6SjdLAkUBuw1Zti1B8hML11#scrollTo=Xa8giaohkT_Q] |
| Ch8 | [https://colab.research.google.com/drive/1qAN-45jk7REhHpIQqS-1rRKuJeWySB5-]  |
| Ch9 | [https://colab.research.google.com/drive/1iGGcN0Wc5b7FwhDRu-5FeNU2lc2_W8Nn] |


##  What We Learned

<h3>Chapter 1,2 and 3 – Exploring and Cleaning Data</h3>

<p>
This chapter demonstrated that data must be properly explored and cleaned before it can be used for machine learning. The learner understood how checking missing values and removing unnecessary columns can improve dataset reliability and recognized that even minor data issues can affect a model’s results.
</p>

<h3>Chapter 4 – Unleashing the Power of Data Through Transformation and Feature Engineering</h3>

<p>
In Chapter 4, the learner gained knowledge about feature engineering, including how to transform data and create new features from existing data. The learner explored basic techniques such as binning, which groups numerical data into different ranges and converts them into categorical data. The learner also understood how to choose appropriate encoding methods based on whether categorical variables have an inherent order or not.
</p>

<h3>Chapter 5 – Data Scaling</h3>

<p>
This chapter demonstrated that features with different numerical ranges can affect how a machine-learning model interprets data. The learner discovered that scaling techniques, such as StandardScaler and MinMaxScaler, can help normalize or standardize feature values, making them more balanced and suitable for machine-learning modeling.
</p>

<h3>Chapter 6 – Dealing with Outliers</h3>

<p>
In Chapter 6, the learner gained an understanding of outliers and how to detect them using Z-scores. The learner also learned that when the Z-score method is not suitable, the Interquartile Range (IQR) method can be used to identify outliers, along with different techniques for handling them. This chapter highlighted how a single outlier can significantly affect a dataset and distort the results of data analysis.
</p>

<h3>Chapter 7 – Feature Selection</h3>

<p>
This chapter demonstrated that not all features are necessary for a machine-learning model and that different methods can identify important features in different ways. The learner discovered that techniques such as Filter, RFECV, and LassoCV can reduce the number of features while retaining the information most useful for making accurate predictions.
</p>

<h3>Chapter 8 – Constructing a Preprocessing Pipeline</h3>

<p>
In Chapter 8, the learner understood that preprocessing pipelines work like conveyor belts, processing raw data through a series of steps until it is ready for use in a machine-learning model. The learner also gained knowledge of how preprocessing pipelines and ColumnTransformers combine different preprocessing steps into a single workflow. This chapter highlighted the efficiency of using pipelines, as they can be reused on new data to produce consistent results while reducing manual work and the possibility of human error.
</p>

<h3>Chapter 9 – Full Pipeline and Evaluation</h3>
The members  gained a better understanding of how data preparation and preprocessing contribute to building an effective machine-learning model. Through the activities, the learner recognized the importance of organizing, transforming, and analyzing data to improve its quality and reliability. These lessons demonstrated that proper data preparation is an essential step in achieving more accurate and consistent machine-learning results.


## Errors we found

List any mistake you found in the original notebooks, and the correct version.
There are real ones in there. Finding them earns points.

<table border="1" style="border-collapse: collapse; width: 100%; table-layout: fixed;">
  <thead>
    <tr>
      <th style="width: 50%; padding: 12px; text-align: center; vertical-align: middle;">
        Mistake in the Original Notebook
      </th>
      <th style="width: 50%; padding: 12px; text-align: center; vertical-align: middle;">
        Correct Version
      </th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td style="padding: 12px; vertical-align: top;">
        The correlation example says
        <strong>“assignments completed ↔ grades unrelated”</strong>
        under zero correlation.
      </td>
      <td style="padding: 12px; vertical-align: top;">
        The actual data shows <strong>Assignments Completed</strong> and
        <strong>Final Grade</strong> have a strong positive correlation
        (<strong>0.917</strong>), so they are clearly related.
      </td>
    </tr>
    <tr>
      <td style="padding: 12px; vertical-align: top;">
        The notebook uses <code>RFECV(cv=5)</code> with only
        <strong>7 data samples</strong>, which produces warnings that R² is
        not well-defined with fewer than two samples in some folds.
      </td>
      <td style="padding: 12px; vertical-align: top;">
        Use a smaller number of folds, such as <code>cv=3</code>, or use a
        larger dataset before applying 5-fold cross-validation.
      </td>
    </tr>
    <tr>
      <td style="padding: 12px; vertical-align: top;">
        The notebook's RFECV result selects only
        <strong>assignments completed</strong>, even though the features are
        highly correlated and the dataset is extremely small.
      </td>
      <td style="padding: 12px; vertical-align: top;">
        This result should be treated cautiously. A larger dataset and
        appropriate cross-validation would give a more reliable
        feature-selection result.
      </td>
    </tr>
    <tr>
      <td style="padding: 12px; vertical-align: top;">
        <strong>Ordinal Encoding Numbering:</strong>
        The markdown says Little = 1, Medium = 2, Lots = 3, but Ordinal
        Encoding outputs are Little = 0, Medium = 1, Lots = 2.
      </td>
      <td style="padding: 12px; vertical-align: top;">
        Either fix the markdown to
        <strong>Little = 0, Medium = 1, Lots = 2</strong>,
        or shift the result by 1:
        <br><br>
        <code>df_3['Ice_encoded'] = ord_enc.fit_transform(df_3[['Ice']]) + 1</code>
      </td>
    </tr>
    <tr>
      <td style="padding: 12px; vertical-align: top;">
        The Z-score note contradicts the output. The markdown says
        <strong>100 is a clear outlier</strong>, but the output is
        <code>Outliers: []</code>.
      </td>
      <td style="padding: 12px; vertical-align: top;">
        The Z-score method did not find any outliers. The z-score of
        <strong>100</strong> is about <strong>2.62</strong>, which is below
        the cutoff of <strong>3</strong>.
      </td>
    </tr>
    <tr>
      <td style="padding: 12px; vertical-align: top;">
        The notes say the result is <strong>“ready for ML models”</strong>,
        but it isn't. <code>ColumnTransformer</code> drops unlisted columns,
        so <code>X_transformed</code> has only scaled Age and Fare.
      </td>
      <td style="padding: 12px; vertical-align: top;">
        <code>X_transformed</code> contains only the scaled
        <strong>Age</strong> and <strong>Fare</strong>. Other columns still
        need encoding.
      </td>
    </tr>
  </tbody>
</table>



## Note on AI tools
- Base on Member 1 (Anne Francine Alonzo) understanding:
I used AI tools to support my learning and complete the activities. I used Gemini to help correct or troubleshoot code when I typed something incorrectly, and I used ChatGPT to help me understand the lessons, clarify concepts, and organize my answers. 

- Base on Member 2 (Charles Edward R. Caraig) understanding:
I used AI tools to aid in understanding the lessons and completing my activities. I used Gemini to aid in troubleshooting code. I used Claude to help me understand the lessons, clarify concepts and codes.


## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
