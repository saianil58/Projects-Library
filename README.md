### Project Library related to AI and ML
![Image](https://res.cloudinary.com/springboard-images/image/upload/q_auto,f_auto,fl_lossy/wordpress/2019/05/aiexcerpt.png)
### Introduction
---

In this self paced collection of Repositories, you will find many projects related to Supervised Learning, Unsupervised Learning, Recommender Systems and Deep Learning that I have worked on so far. Throughout this guideline you will walk through the details of projects and repositories.

You may reach me whenever you want to get further information about projects. 

### Index
|__Problem__|__Methods__|__Libs__|__Repo__|
|-|-|-|-|
|[Real Time Analytics using SPARK](#Real-Time-Analytics-using-SPARK)|`Spark Streaming`, `Twitter`, `Real Time Analytics` |`StreamListener`, `Spark`|[Click](https://github.com/saianil58/ML-with-SPARK/tree/master/RTA)|
|[Model Interpretability](#Model-Interpretability)|`Interpretation`, `Deep Learning`, `pipeline` |`SHAP`, `ELI5`, `LIME`, `XGB`, `Random Forests`|[Click](https://nbviewer.jupyter.org/github/saianil58/Model-Interpretability/blob/master/Introduction%20to%20Model%20Interpretability.ipynb)|
|[Tensorflow Keras Model Deployment](#Tensorflow-Keras-Model-Deployment)|`Deployment`, `Deep Learning` |`Keras`, `Flask`, `PIL`, `base64`|[Click](https://github.com/saianil58/Keras-Model-Deployment)|
|[Telecom Churn Classification using Spark](#Telecom-Churn-Classification-using-Spark)|`Classification`, `GBTClassifier`, `RandomForestClassifier`, `DecisionTreeClassifier` |`pyspark`, `koalas`, `mllib`|[Click](https://nbviewer.jupyter.org/github/saianil58/ML-with-SPARK/blob/master/Churn%20Classification%20Spark.ipynb)|
|[Fashion MNIST Classification](#Fashion-MNIST-Classification)|`Classification`, `DeepLearning`, `ANN` |`Keras`, `plotly`|[Click](https://nbviewer.jupyter.org/github/saianil58/Artificial-Neural-Networks/blob/master/Image%20Classifications/Fashion_images.ipynb)|
|[Bank Customer Churn Modelling using Neural Networks](#Bank-Customer-Churn-Modelling-using-Neural-Networks)|`Classification`, `DeepLearning`, `ANN`, `Statistical Tests` |`Keras`, `AutoViz`, `plotly`, `pandas`, `seaborn`, `TALOS`|[Click](https://nbviewer.jupyter.org/github/saianil58/Artificial-Neural-Networks/blob/master/Binary%20Classification/Bank%20Churn%20Prediction%20using%20ANN.ipynb)|
|[HR Analytics and classification using ANN](#HR-Analytics-and-classification-using-ANN)|`Classification`, `DeepLearning`, `ANN` |`Keras`, `AutoViz`, `plotly`, `pandas`, `seaborn`,`StratifiedKFold`|[Click](https://nbviewer.jupyter.org/github/saianil58/Artificial-Neural-Networks/blob/master/Binary%20Classification/binary_classification_keras.ipynb)|
|[ML Model Deployment](#ML-Model-Deployment)|`Deployment` |`pickle`, `flask`, `ensemble`, `pandas`|[Click](https://github.com/saianil58/Model_Deployment)|
|[Clustering cars based on attributes](#Clustering-cars-based-on-attributes)|`UnSupervised learning` |`plotly`, `AgglomerativeClustering`, `K-Means`, `Dendograms`, `silhouette`|[Click](https://nbviewer.jupyter.org/github/saianil58/Unsupervised-Learning/blob/master/Cars%20Clustering%20.ipynb)|

Please, scroll down to see the details of projects comprehensively and visit their repository.

### Real Time Analytics using SPARK 
---
Real Time Analytics is a growing feild where data would be streaming continously and analysis would be done on the data.
![Image](https://www.mentionlytics.com/wp-content/uploads/2020/01/Nine-Free-Hashtag-Tracking-Tools-to-Use-for-Higher-Post-Reach.jpg)
### Model Interpretability
---

## Please Explain your Predictions?

Machine learning is in the center of the latest progress in technology and is an essential tool for accurate predictions nowadays. However, most of the time we neither can clearly identify nor explain the logic behind these predictions because the model is just too complex. In those cases our machine learning model is called a ’Black Box’.

![Image](https://d33wubrfki0l68.cloudfront.net/5331cb13d71df10783ce7b69c7bc9f703db5bf3d/2ecd6/img/posts/lime/intro.png)

So how do we know if we can trust this model? How should we be able to trust it, when we don’t even know how it actually makes it’s predictions?

These are important questions which occur when the challenges of Model explainability are presented, especially if it is used for decision making. Users need to be confident that the model will perform well. Gaining trust in predictions through increasing transparency of a black box model.

There are multiple python libraries for this task.
1. ELI5
2. LIME
3. SHAP

Practical classification task with mutliple blackbox models and their interpretations are in the [Repo](https://nbviewer.jupyter.org/github/saianil58/Model-Interpretability/blob/master/Introduction%20to%20Model%20Interpretability.ipynb)

More details are [here](https://github.com/saianil58/Model-Interpretability/blob/master/README.md)

### Tensorflow Keras Model Deployment
---

Often there’s a need to abstract away your machine learning model details and just deploy or integrate it with easy to use API endpoints. For eg., We can provide a URL endpoint using which anyone can make a POST request and they would get a JSON response of what the model has inferred without having to worry about its technicalities.

In this project, we will create a Flask server to deploy our MSIST classification artificial neural network (ANN) built in Keras. We will then create a simple Flask server which will accept POST request and do some image preprocessing and return the predictions.

Server code is [Here](https://github.com/saianil58/Keras-Model-Deployment/blob/master/Server_File.py)

Model building and saving is [Here](https://github.com/saianil58/Keras-Model-Deployment/blob/master/Classification_MNIST_ANN_Keras.ipynb)

For the data input of image we are using CSS here but we can use any technology as long as we call the API on our server in specifed path and required format.

CSS source code is [Here](https://github.com/saianil58/Keras-Model-Deployment/tree/master/static)

After the server is launced and we can go to specified url, in this case as we are doing local deployment the URL would be "http://127.0.0.1:8111/"

![Image](https://github.com/saianil58/Keras-Model-Deployment/blob/master/images/server.JPG)

The screen on that url would look like this:
![Image](https://github.com/saianil58/Keras-Model-Deployment/blob/master/images/one.JPG)

We just need to write any number in the white box and click on predict button and we would see the result.
![Image](https://github.com/saianil58/Keras-Model-Deployment/blob/master/images/two.JPG)



### Telecom Churn Classification using Spark
---
![Image](https://miro.medium.com/max/3384/1*WqId29D5dN_8DhiYQcHa2w.png)

Customer Churn is a bigger problem for most of the companies. If we can identify the potential customers who could be leaving the company, we can take precautionary measures and retain the customers. This project deals with Telecom data and uses pyspark which helps in working with larger datasets in organizations with massive data.

![Image](https://www.edureka.co/blog/wp-content/uploads/2018/07/PySpark-logo-1.jpeg)
Spark, is a framework based on Hadoop technologies that provide far more flexibility and usability than traditional Hadoop. It is probably the best tool for managing and analyzing large datasets, aka Big Data.

The Spark ML framework allows developers to use Spark for data processing at scale while building machine learning models.

[Here is a notebook](https://nbviewer.jupyter.org/github/saianil58/ML-with-SPARK/blob/master/Churn%20Classification%20Spark.ipynb) to predict churn from telecom data

I have used many concepts like
1. Koalas
2. Preprocessing in Spark
3. Data Visualisation
4. Building pipelines and generating Feature importances
5. Compare different models
6. Insights learned from the whole process

### Fashion MNIST Classification
---

### Bank Customer Churn Modelling using Neural Networks
---
![Image](https://orzota.com/wp-content/uploads/2014/04/churn_prediction_with_Dato.png)
#### A Definition of Customer Churn

Simply put, customer churn occurs when customers or subscribers stop doing business with a company or service. Also known as customer attrition, customer churn is a critical metric because it is much less expensive to retain existing customers than it is to acquire new customers – earning business from new customers means working leads all the way through the sales funnel, utilizing your marketing and sales resources throughout the process. Customer retention, on the other hand, is generally more cost-effective as you’ve already earned the trust and loyalty of existing customers.

Customer churn impedes growth, so companies should have a defined method for calculating customer churn in a given period of time. By being aware of and monitoring churn rate, organizations are equipped to determine their customer retention success rates and identify strategies for improvement.

![Image](https://data-flair.training/blogs/wp-content/uploads/sites/2/2019/07/Introduction-to-Artificial-Neural-Networks-1280x720.jpg)
This Kaggle project involves building an ANN-based churn model which can determine whether certain bank customers will continue using their service or not. The ANN model analyzes the relationship between customer churn & multiple independent variables affecting churn. Recommendations for improvements in service were suggested based on the results of the analysis.

#### Skills and Tools
Neural Networks, Classification, Keras, Tensorflow

### HR Analytics and classification using ANN
---

### ML Model Deployment
---

### Clustering cars based on attributes
---
Analyzed cars dataset and performed exploratory data analysis and then categorized them using K means clustering. Used linear regression on the different clusters and estimated coefficients.

#### Skills and Tools
K means clustering, Hierarchical clustering, EDA, Linear Regression
