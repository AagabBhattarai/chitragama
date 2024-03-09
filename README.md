Welcome to the Chitragama Structure from Motion project.

This is a simple project that attempts to complete the construction of sparse 3D point cloud from a given list of images taken with same camera.

This version of software was lastly used with python 3.11, so newer or older versions of python may not fulfill all the dependencies

To use this, simply first clone this repository in a directory of your choice.

Then it is recommended you create a python virtual environment by the command as follow
``` python -m venv \path\to\venv\ ```
where \path\to\venv\ is any folder in of choice where to create a python virtual environment

Now, if you used a virtual environment, activate the virtual environment in a shell context by
``` \path\to\venv\Scripts\activate ```

Next, install the requirements using the requirements.txt in the python environment
``` pip install -r requirements.txt ```

Finally launch the chitragama app by
``` python main_app.py ```
