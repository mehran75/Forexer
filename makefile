export USERNAME = "<Username>"
export PASSWORD = "<Password>"
export configuration_file = "configuration/parameters_eurusd.yml"

# check if requirements exists
# install if not

.PHONY: all
run:
	python main.py $(configuration_file)