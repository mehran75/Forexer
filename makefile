export apikey = "apikey from https://www.alphavantage.co/"
export configuration_file = "configuration/parameters.yml"

# check if requirements exists
# install if not
# todo: install packages


run:
	python main.py $(configuration_file)
