import pandas as pd
import matplotlib.pyplot as plt

def importer():
	red_data = pd.read_csv('winequality-red.csv',sep=';')
	white_data = pd.read_csv('winequality-white.csv',sep=';')

	return (red_data, white_data)

def main():
	red_data, white_data = importer()

if __name__ == "__main__":
	main()