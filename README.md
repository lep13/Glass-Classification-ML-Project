# Glass-Classification-ML-Project
This project aims to classify glass samples using machine learning techniques implemented in a Jupyter notebook, by analyzing various physical and chemical properties of the glass to accurately predict its type.

"Glass Identification" data set from the UCI Machine Learning Repository: This data set contains 214 samples of glass of different types and their respective refractive index and chemical composition. The data set can be found at: https://archive.ics.uci.edu/ml/datasets/Glass+Identification


# OVERVIEW
Developing novel glasses with new, improved properties and functionalities is key
to address some of the Grand Challenges facing our society. Although the
process of designing a new material is always a difficult task, the design of novel
glasses comes with some unique challenges. First, virtually all the elements of the
periodic table can be turned into a glass if quenched fast enough. Second, unlike
crystals, glasses are intrinsically out-of-equilibrium and, hence, can exhibit a
continuous range in their stoichiometry (within the glass-forming ability domain).
For these reasons, the compositional envelope that is accessible to glass is
limitless and, clearly, only an infinitesimal fraction of these compositions have
been explored thus far.
As a first option for glass classification, physics-based modelling can greatly
facilitate the design of new glasses by predicting a range of optimal promising
compositions to focus on . For instance, topological constraint theory has led to
the development of several analytical models predicting glass properties as a
function of their compositions (e.g., glass transition temperature, hardness,
stiffness, etc.) . However, the complex, disordered nature of glasses renders
challenging the development of accurate and transferable physics-based models
for certain properties (e.g., liquidus temperature, fracture toughness, dissolution
kinetics, etc.) . Alternatively, “brute-force” atomistic modelling techniques (e.g.,
molecular dynamics) can be used to accurately compute glass properties and
partially replace more costly experiments. However, such techniques come with
their own challenges (e.g., limited timescale, small number of atoms, fast cooling
rate, large computing cost, etc.), which prevents a systematic exploration of all
the possible glasses.
As an alternative route to physics-based modelling, artificial intelligence and
machine learning offer a promising path to leverage existing datasets and infer
data-driven models that, in turn, can be used to accelerate the discovery of novel
glasses. In details, machine learning can “learn from example” by analyzing
existing datasets and identifying patterns in data that are invisible to human eyes
. First, some data are generated (by experiments, simulations, or mining from
existing databases) to build a database of properties. Such databases can
comprise, as an example, the glass composition, synthesis procedure, as well as
select properties.
Machine learning is then used to infer some patterns within the dataset and
establish a predictive model.
Machine learning algorithms can accomplish two types of tasks, namely,
supervised and unsupervised. In the case of supervised machine learning, the
dataset comprises a series of inputs (e.g., glass composition) and outputs (e.g.,
density, hardness, etc.). Supervised machine learning can then learn from these
existing examples and infer the relationship between inputs and outputs.
Supervised machine learning comprises (i) regression algorithms, which can be
to predict the output as a function of the inputs (e.g., composition-property

predictive models) and (ii) classification algorithms, which can be used to label
glasses within different categories. In contrast, in the case of unsupervised
machine learning, the dataset is not labeled (i.e., no output information is known).
Unsupervised machine learning can, for instance, be used to identify some
clusters within existing data, that is, to identify some families of data points that
share similar characteristics.

# METHOD
1. Import the required modules like pandas, numpy, etc.
2. Import the glass data frame as cvs file.
3. Learn the basic characteristics of the data frame such as its size, number of
NaN values,etc.
4. To check the correlation between each columns create a heat map using the
seaborn module.
5. Then split the columns into X and Y values. Y being the dependant variable that
is the glass type(i.e.1,2,3,4,5,6,7). X values consist of RI,Na,K,etc.
6. Split the x-values and y-values into training and testing datasets.
7. Import the kneighnoursclassifier and look for the best value of k.
8. We find the best value of k to be 1, hence we go ahead and train the model
using the KNN classifier with k=1.
9. After training the model with the training dataset, we predict the values of the
testing dataset.
10. The predicted y-values(i.e.1,2,3,4,5,6,7) will be compared with the actual
values(y_test) using a confusion matrix. We are then provided with an accuracy
score based on the confusion matrix.
11. We get an accuracy rate of 75% with this model.
12. Now, to increase the accuracy rate, we oversample the skewed data to even
out the glass type distribution.
13. Trying the same method to this new balanced dataframe, we get an accuracy
rate of 89%.
This we can predict the glass type with a high accuracy rate.

KNN
KNN makes predictions using the training dataset directly. Predictions are made
for a new instance (x) by searching through the entire training set for the K most
similar instances (the neighbors) and summarizing the output variable for those
K instances. For regression this might be the mean output variable, in
classification this might be the mode (or most common) class value.
To determine which of the K instances in the training dataset are most similar to
a new input a distance measure is used. For realvalued input variables, the most
popular distance measure is Euclidean distance.
Euclidean distance is calculated as the square root of the sum of the squared
differences between a new point (x) and an existing point (xi) across all input
attributes j.
EuclideanDistance(x, xi) = sqrt( sum( (xj – xij)^2 ) )
Other popular distance measures include:
Hamming Distance: Calculate the distance between binary vectors (more).
Manhattan Distance: Calculate the distance between real vectors using the sum
of their absolute difference. Also called City Block Distance (more).
Minkowski Distance: Generalization of Euclidean and Manhattan distance (more).
There are many other distance measures that can be used, such as Tanimoto,
Jaccard, Mahalanobis and cosine distance. You can choose the best distance
metric based on the properties of your data. If you are unsure, you can
experiment with different distance metrics and different values of K together and
see which mix results in the most accurate models.
Euclidean is a good distance measure to use if the input variables are similar in
type (e.g. all measured widths and heights). Manhattan distance is a good
measure to use if the input variables are not similar in type (such as age, gender,
height, etc.).
The value for K can be found by algorithm tuning. It is a good idea to try many
different values for K (e.g. values from 1 to 21) and see what works best for your
problem.
The computational complexity of KNN increases with the size of the training
dataset. For very large training sets, KNN can be made stochastic by taking a
sample from the training dataset from which to calculate the K-most similar
instances.

# LIMITATIONS AND CHALLENGES:
Although machine learning offers a unique, largely untapped opportunity to
accelerate the discovery of novel glasses with exotic functionalities, it faces
several challenges.
First, the use of machine learning requires as a prerequisite the existence of data
that are
(i)available (i.e., public and easily accessible)
(ii)complete
(iii)consistent (e.g., obtained from a single operation)
(iv)accurate (i.e., with low error bars), and
(v)numerous
For instance, although some glass property databases are available,
inconsistencies between data generated by different groups render challenging
the meaningful application of machine learning approaches.
In addition, since they are usually only driven by data and do not embed any
physics- or chemistry-based knowledge, machine learning models can
sometimes violate the laws of physics or chemistry.

# OVERVIEW OF MACHINE LEARNING TECHNIQUES FOR GLASS SCIENCE REGRESSION TECHNIQUES

1. PARAMETRIC AND NONPARAMETRIC REGRESSION
Regression consists of fitting known data points to establish a functional
relationship between the inputs and output. A regression models are able to
interpolate known points by learning from an existing dataset. Generally,
regression methods can be categorised into

(i)parametric regression, which yields an analytical formula expressing the output
in terms of the input variables (e.g., linear, polynomial, or nonlinear functions) and
(ii)nonparametric regression, which defines a kernel function to calculate the
output at a given input position based on the correlation between this input
position and its surrounding known points.

2. OPTIMIZATION OF MODEL COMPLEXITY
The development of supervised learning models usually comprises two stages,
(i)the learning/fitting (i.e., training and validation) stage and
(ii)the prediction (i.e., test) stage.
• Learning and fitting stage: During the fitting/learning stage, it is key to properly
adjust the complexity of the model (e.g., the maximum degree in polynomial
regression) to offer reliable predictions.
• Prediction stage: Once the optimal degree of complexity is fixed, the test set is
used to assess the accuracy of the model by comparing its predictions to a
fraction of the dataset that is kept unknown to the model.

# MACHINE LEARNING FORCEFIELDS FOR GLASS MODELLING
MD simulations are an important tool to access the atomic structure of glasses
and, thereby, decipher the nature of the relationship between glass composition
and properties. However, the reliability of MD (or Monte Carlo) simulations is
intrinsically limited by that of the interatomic forcefield that is used, which acts as
a bottleneck in glass modelling. To this end, machine learning offers a promising
route to develop new accurate interatomic forcefields for glass modelling in an
efficient and non-biased fashion. Although various studies have focused on the
use of machine learning to develop complex, non-analytical interatomic
forcefields, such forcefields present low interpretability and have been largely
restricted to simple monoatomic or diatomic systems thus far.

# CONCLUSION
Overall, machine learning techniques offer a unique, largely untapped
opportunity to leapfrog current glass design approaches—a process that has thus
far remained largely empirical and based on previous experience. When
combined with physics-based modeling, machine learning can efficiently and
robustly interpolate and extrapolate predictions of glass properties as a function
of composition and, hence, drastically accelerate the discovery of new glass
formulations with tailored properties and functionalities. It is worth pointing out
that, when adopting machine learning, different properties may come with
different challenges and different degrees of complexity. Various criteria can be
used to describe the complexity of a given property
