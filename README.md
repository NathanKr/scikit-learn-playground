<h2>Motivation</h2>
python code to do basicscikit-learn

<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>boston_housing_linear_regression.py</td>
    <td>
    <ul>
    <li>use sklearn.linear_model.LinearRegression with 1 order ploynomial as model to predict house price in boston year 1978</li>
    <li>plot the features</li>
    <li>plot the learning curves. it is obvious that we have high bias problem. so higher order ploynomial might be better or maybe other model</li>
    <li>open issues</li>
    <ul>
    <li>some of the points on the learning curves are very negative so i had to use plt.ylim . whats the reason for this ?</li>
    <li>what data cleaning \ pre processing can be used to improve the model ?</li>
    <li>why do i get noisy learning curve - in particular the one for test ?</li>
    <li>use the learning_curve function from sklearn.model_selection</li>
    </ul>
    </ul>
    </td>
  </tr>
  <tr>
    <td>pre_processing.py</td>
    <td>
    <ul>
    <li>Standardize features by removing the mean and scaling to unit variance according to (x - u) / s where u is the mean and s is standard deviation</li>
    </ul>
    </ul>
    </td>
  </tr>
</table>
