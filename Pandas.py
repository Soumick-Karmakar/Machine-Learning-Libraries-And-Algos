from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)


################## Series ##################

labels = ['a', 'b', 'c']
my_data = [10, 20, 30]
arr = np.array(my_data)
dic = {'a': 10, 'b': 20, 'c': 30}
# print(pd.Series(my_data))
# print(pd.Series(my_data,labels))
# print(pd.Series(labels,my_data))
# print(pd.Series(data = my_data, index = labels))
# print(pd.Series(arr,labels))
# print(pd.Series(dic))
# print(pd.Series(data=[sum,print,len]))

country1 = pd.Series([1, 2, 3, 4], ['India', 'Russia', 'Japan', 'USA'])
# print(country1)

country2 = pd.Series([1, 2, 3, 4], ['India', 'Russia', 'Japan', 'Portugal'])
# print(country2)

# print(country1 + country2)  #will add the numbers set for the countries


################# Data Frame ####################

df = pd.DataFrame(randn(5, 4), ['a', 'b', 'c', 'd', 'e'], ['w', 'x', 'y', 'z'])
# print(df)
# print(type(df['w']))
# print(type(df))
# print(df[['w','z']])
df['new'] = df['w'] + df['z']  # this will add a new column to the DataFrame
# print(df)
df.drop('new', axis=1, inplace=True)  # axis = 0 [rows]  || axis = 1 [columns]
# print(df)

# Note:
#   inplace = true is given to make the changes, if we would not have
#   written the inpalce = true, then we would have to assign the df.drop
#   command to another variable to save the changes made to the df
#
#   also for axis option, axis = 0 is default, but for axis = 1 we need
#   to mention the option

# print(df.loc['a']) #this will take row name and display that particular row
# print(df.iloc[0])  #this will take row index and display that particular row
# print(df.loc['b','x']) #this will take row and column name respectively and print cell value
# print(df.loc[['a','c'],['w','z']]) #to return subsets of the dataframe


# Conditional Selection
# print(df > 0) #this will return a table of true/false for the condition df > 0
# num = df > 0
# print(df[num]) #this will return the values if the condition is true otherwise NaN
# print(df['w'] > 0) # this will return true/false for that particular column
# print(df[df['w'] > 0]) #this will remove the entire row of the table for which the row value of w column is false
# print(df[(df['w']>0) & (df['y']>1)])
# print(df[(df['w']>0) | (df['y']>1)])
# print(df.reset_index())


newind = "CA NY WY OR CO".split()
# df['States'] = newind # to add a new column named States
# print(df)
# df.set_index('States') # to add a column of index with values of newind as index values
# print(df)


# Index Levels
outside = "G1 G1 G1 G2 G2 G2".split()
inside = [1, 2, 3, 1, 2, 3]
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

# creating multi index level DataFrame
df = pd.DataFrame(randn(6, 2), hier_index, ['A', 'B'])
# print(df)
# print(df.loc['G1'].loc[1]) # to get datas from multilevel index
# print(df)
# this is done to fill the empty heading areas
df.index.names = ['Groups', 'Num']
# print(df)

# print(df.loc['G1'])
# print(df.xs('G1')) # xs is cress section which works similar to loc as given

# xs is useful where I want to get all data from Num = 1 irrespective of any groups
# print(df.xs(1,level='Num'))


########## Missing Data ################

d = {'A': [1, 2, np.nan], 'B': [5, np.nan, np.nan], 'C': [1, 2, 3]}
df = pd.DataFrame(d)
# print(df)

# print(df.dropna(axis=1)) # This drops all columns that contains N/A values
# print(df.dropna(thresh=2)) # thresh meams threshould value. Any row that contains N/A values >= that number only those rows will be dropped.
# print(df.dropna(subset=['A'])) # This will drop all rows which will have value NaN for column 'A'.

# print(df.fillna(value='Fill Val')) # tghis will fill the N\A values with "Fill Val" comment
# print(df['A'].fillna(value=df['A'].mean())) # to fill value with mean fo the column


############### Groupby ####################

data = {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'], 'Person': [
    'Sam', 'Charlie', 'Amy', 'Jordan', 'Carl', 'Sarah'], 'Sales': [200, 120, 340, 124, 242, 350]}
df = pd.DataFrame(data)

group_by_Company = df.groupby('Company')
# print(group_by_Company['Sales'].mean())
# print(group_by_Company['Sales'].sum())
# print(group_by_Company['Sales'].std())
# print(group_by_Company['Sales'].var())

# print(group_by_Company['Sales'].sum().loc['FB']) # this will on;y return the sum for FB
# print(group_by_Company.count())
# print(group_by_Company.max())
# print(group_by_Company.min())
# print(group_by_Company.describe()) #this will print all values like max min std, etc
# print(group_by_Company.describe().transpose()) # row and column will be interchanged


################## Merging, Joining, and Concatenating ################

### Concatenation ###
df1 = pd.DataFrame({'A': ['a0', 'a1', 'a2', 'a3'], 'B': ['b0', 'b1', 'b2', 'b3'], 'C': [
                   'c0', 'c1', 'c2', 'c3'], 'D': ['d0', 'd1', 'd2', 'd3']}, index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['a4', 'a5', 'a6', 'a7'], 'B': ['b4', 'b5', 'b6', 'b7'], 'C': [
                   'c4', 'c5', 'c6', 'c7'], 'D': ['d4', 'd5', 'd6', 'd7']}, index=[4, 5, 6, 7])
df3 = pd.DataFrame({'A': ['a8', 'a9', 'a10', 'a11'], 'B': ['b8', 'b9', 'b10', 'b11'], 'C': [
                   'c8', 'c9', 'c10', 'c11'], 'D': ['d8', 'd9', 'd10', 'd11']}, index=[8, 9, 10, 11])
# print(pd.concat([df1,df2,df3])) # will merge the 3 dataframes
# print(pd.concat([df1,df2,df3],axis=1)) # concatenating along axis


### Merging ###
left = pd.DataFrame({'key': ['k0', 'k1', 'k2', 'k3'], 'A': [
                    'a0', 'a1', 'a2', 'a3'], 'B': ['b0', 'b1', 'b2', 'b3']})
right = pd.DataFrame({'key': ['k0', 'k1', 'k2', 'k3'], 'C': [
                     'c0', 'c1', 'c2', 'c3'], 'D': ['d0', 'd1', 'd2', 'd3']})
# print(pd.merge(left,right,how='outer',on='key'))

left1 = pd.DataFrame({'key1': ['k0', 'k1', 'k2', 'k3'], 'key2': [
                     'k0', 'k1', 'k0', 'k1'], 'A': ['a0', 'a1', 'a2', 'a3'], 'B': ['b0', 'b1', 'b2', 'b3']})
right1 = pd.DataFrame({'key1': ['k0', 'k1', 'k2', 'k3'], 'key2': [
                      'k0', 'k0', 'k0', 'k0'], 'C': ['c0', 'c1', 'c2', 'c3'], 'D': ['d0', 'd1', 'd2', 'd3']})
# print(pd.merge(left1,right1, on=['key1', 'key2']))
# print(pd.merge(left1,right1, how='outer' ,on=['key1', 'key2']))
# print(pd.merge(left1,right1, how='inner' ,on=['key1', 'key2']))
# print(pd.merge(left1,right1, how='right' ,on=['key1', 'key2']))
# print(pd.merge(left1,right1, how='left' ,on=['key1', 'key2']))


### Joining ###
left2 = pd.DataFrame({'A': ['a0', 'a1', 'a2'], 'B': [
                     'b0', 'b1', 'b2']}, index=['k0', 'k1', 'k2'])
right2 = pd.DataFrame({'C': ['c0', 'c1', 'c2'], 'D': [
                      'd0', 'd1', 'd2']}, index=['k0', 'k2', 'k3'])
# print(left2.join(right2))
# print(left2.join(right2, how='outer'))


##################### Operations #####################

df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [
                  444, 555, 666, 444], 'col3': ['abc', 'def', 'ghi', 'xyz']})
# print(df.head())
# print(df['col2'].unique()) # will get all the unique values only
# print(df['col2'].nunique()) # will return the length of the array, alternatively we can also use the len func
# print(len(df['col2'].unique())) # this will also work the same like above nunique func
# print(df['col2'].value_counts()) # this will return the count of the unique values
# print(df[df['col1']>2]) # will return the rows where col1 values are greater than 1
# print(df[(df['col1']>2) & (df['col2']==444)])
# print(df[(df['col1']>2) | (df['col2']==444)])

df = pd.DataFrame([[1,'Soumick'],[2,'Lana'],[3,'Soumick']],columns=['EmpID','Name'])
df.drop_duplicates(subset='Name', keep='first', inplace=True) # this will remove the duplicates from the Name column and keep the 1st occurrence only
# print(df)


def times2(x):
    return x*2
# print(df['col1'].apply(times2)) # apply operator takes one function and applies over a selected row/col
# print(df['col3'].apply(len))
# print(df['col2'].apply(lambda x: x*2))

# print(df.drop('col1',axis=1)) # to delete columns from table
# print(df.columns) # this will display the columns names
# print(df.index)
# print(df.sort_values('col2')) # sorting data frame with the respective column name
# print(df.isnull()) # will return boolean values


data = {'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'], 'B': ['one', 'one', 'two',
                                                               'two', 'one', 'one'], 'C': ['x', 'y', 'x', 'y', 'x', 'y'], 'D': [1, 3, 2, 5, 4, 1]}
df = pd.DataFrame(data)
# print(df.pivot_table(values="D",index=['A','B'],columns=['C']))


################ Data Input and Output #########################

# conda install sqlalchemy
# conda install lxml
# conda install html5lib
# conda install BeautifulSoup4

# df = pd.read_csv('example.csv') # to read form csv
# df.to_csv('My_output',index=False)

# df = pd.read_excel('Excel_Sample.xlsx',sheetname='Sheet1') # reading from excel file
# df.to_excel('Excel_Sample2.xlsx',sheet_name='NewSheet')

# data = pd.read_html('http://127.0.0.1:5500/index.html') # this accepts online html page link
# print(data[0].head())

# this will create a very small temporary sqlite engine db
engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table', engine)
sqldf = pd.read_sql('my_table', con=engine)
# print(sqldf)
