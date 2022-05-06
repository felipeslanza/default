[default]
=======
This projects provides an API to estimate the probabiltiy of default of a given user.


Requirements
------------

* `python >= 3.9`


Setup
-----
Simply clone this repository and setup an environment:

```shell
# Clone the repo
git clone git@github.com:felipeslanza/default.git

# Setup environment and install dependencies
cd default
virtualenv -p python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```


Using the API
-------------
To use the API, simply send a `POST` request whose body contains valid `JSON` with the
user's data. Just make sure you add a `"Content-Type: application/json"` header to the
request, e.g.:

```shell
curl -X POST https://v11o0q9hbe.execute-api.sa-east-1.amazonaws.com/dev/api \
     -H "Content-Type: application/json" \
     -d "@data/sample_user.json"
```
