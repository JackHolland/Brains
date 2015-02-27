import pandas as pd
import sys

def extract_data(filename):
	with open(filename, 'r') as f:
		lines = f.read().splitlines(True)[1:]
		data = []
		for line in lines:
			cells = line.split(',')
			l_data = [i.strip() for i in cells]
			data.append(l_data)
		df = pd.DataFrame(data,columns = ['gene','label', 'P', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32', '34', '36', '38', '40', '42', '44', '46', '48'])
	return df

def subset_data(pddf):
	train = pddf[pd.concat([pddf['label'] == '1', pddf['label'] == '-1'], axis=1).any(axis=1)].copy(deep=True)
	test = pddf[pddf['label'] == '0'].copy(deep=True)
	train['label'][train['label'] == '-1'] = '0'
	test['label'][test['label'] == '0'] = 'None' 
	train.to_csv('train.csv',index = False)
	test.to_csv('test.csv',index = False)
	return



if __name__ == '__main__':
	in_file = sys.argv[1]
	extracted = extract_data(in_file)
	subset_data(extracted)

			