import sys
import numpy as np
import nltk as n
import matplotlib.pyplot as plt
import pandas as pd
#from wine import wine as w
#sys.path.append('/Users/aztlan/work/data_science')
#from sets import Set
#data = open('winemag-data-130k-v2.csv','r')

df = pd.read_csv('winemag-data-130k-v2.csv')
dscs = df['description']

#reviews = data.readlines()

count_cherry=0
count_apple=0
count_papple=0
count_lem=0
count_pear=0
count_sberry=0

count_sweet=0
count_sour=0
count_salty=0
count_bitter=0
count_spicy=0
count_umami=0

count_choc=0
count_tea=0
count_coff=0
count_earth=0

flavors = ['sweet','sour','salty','bitter','spicy','umami']
fruits = ['apple','pineapple','lemon','strawberry','cherry','pear','orange','citrus','black berry','grapefruit']
darkf = ['coffee','chocolate','tea','earth']

#fruit_hash = hash(fruits)
fruit_set = set(fruits)
darkf_set = set(darkf)
flavors_set = set(flavors)

for dsc in dscs:

  words = n.word_tokenize(dsc) 

  tmp_word_set = set(words)
  tmp_word_set.intersection_update(fruit_set)
  #fruits
  if fruits[0] in tmp_word_set:
    count_apple=count_apple+1
  if fruits[1] in tmp_word_set:
    count_papple=count_papple+1
  if fruits[2] in tmp_word_set:
    count_lem=count_lem+1
  if fruits[3] in tmp_word_set:
    count_sberry=count_sberry+1
  if fruits[4] in tmp_word_set:
    count_cherry=count_cherry+1
  if fruits[5] in tmp_word_set:
    count_pear=count_pear+1
  #break
  tmp_word_set = set(words)
  tmp_word_set.intersection_update(darkf_set)

  if darkf[0] in tmp_word_set:
    count_coff=count_coff+1
  if darkf[1] in tmp_word_set:
    count_choc=count_choc+1
  if darkf[2] in tmp_word_set:
    count_tea=count_tea+1
  if darkf[3] in tmp_word_set:
    count_earth=count_earth+1

  tmp_word_set = set(words)
  tmp_word_set.intersection_update(flavors_set)

  if flavors[0] in tmp_word_set:
    count_sweet=count_sweet+1
  if flavors[1] in tmp_word_set:
    count_sour=count_sour+1
  if flavors[2] in tmp_word_set:
    count_salty=count_salty+1
  if flavors[3] in tmp_word_set:
    count_bitter=count_bitter+1
  if flavors[4] in tmp_word_set:
    count_spicy=count_spicy+1
  if flavors[5] in tmp_word_set:
    count_umami=count_umami+1


print(f'apples are mentioned {count_apple} times')
print(f'pineapples are mentioned {count_papple} times')
print(f'lemons are mentioned {count_lem} times')
print(f'strawberries are mentioned {count_sberry} times')
print(f'cherries are mentioned {count_cherry} times')
print(f'pears are mentioned {count_pear} times')

fruit_count = [count_apple,count_papple,count_lem,count_sberry,count_cherry,count_pear]
index = [1,2,3,4,5,6]

plt.bar(index,fruit_count)
plt.xticks(index,('apples','papple','lemon','sberry','cherry','pear'))
plt.show()

darkf_count = [count_coff,count_choc,count_tea,count_earth]
index = [1,2,3,4]

plt.bar(index,darkf_count)
plt.xticks(index,('coffee','chocolate','tea','earth'))
plt.show()

flavor_count = [count_sweet,count_sour,count_salty,count_bitter,count_spicy,count_umami]
index = [1,2,3,4,5,6]

plt.bar(index,flavor_count)
plt.xticks(index,('sweet','sour','salty','bitter','spicy','umami'))
plt.show()

