# Spam Classifier Flask App (MLOps Deployment)

A Flask-based REST API that classifies SMS text as 'spam' or 'not spam' using Logistic Regression. Includes Dockerization and optional hyperparameter tuning via `GridSearchCV`.

## Docker Commands

Build and run locally:
-> docker build -t jayesh/spam-classifier-app .
-> docker run -p 5000:5000 jayesh/spam-classifier-app

Push to DockerHub::
-> docker tag jayesh/spam-classifier-app mt24aai032/mlops:FlaskApp
-> docker push mt24aai032/mlops:FlaskApp

Pull from DockerHub and run locally:
docker pull mt24aai032/mlops:FlaskApp 
docker run -p 5000:5000 mt24aai032/mlops:FlaskApp


ENDPOINTS::

"/train": Train the spam classifier model (POST)
	- w/o body  : trains on hyperparameterised model
	- with body : provide params to train on

	
	Content-Type: "application/json"
	Body: {"C": 10,"penalty": "l2","solver": "liblinear"}

"/get_params": "Get last trained model hyperparameters (GET),

"/evaluate": "Predict if input text is spam or not (POST)

	Content-Type: "application/json"
	Body: {"text": "Congratulations! You've won a free iPhone. Click here to claim now!"}


        Sample o/p: {"input_text": "Congratulations! You've won a free iPhone. Click here to claim now!","prediction": "spam"}
       

