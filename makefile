export USERNAME = "<Username>"
export PASSWORD = "<Password>"
export configuration_file = "configuration/parameters_eurusd.yml"

# todo: install packages
run:
	python main.py $(configuration_file)
