
# coding: utf-8

# In[5]:


sizes = [10000,     50000,
         100000,    500000,
         1000000,   1000000,
         10000000,  50000000,
         100000000, 500000000,
         1000000000, 2000000000, 3000000000]    


# In[6]:


def randomLetter(last):
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    
    rand = letters[np.random.randint(len(letters))]
    
    while (rand == last):
        rand = letters[np.random.randint(len(letters))]
    
    return rand


# In[7]:


import numpy as np


# In[ ]:


for size in sizes:
    i = 0
    text = ""
    compressed = ""
    
    lastletter = 'Z'
    while i<size:
        length = np.random.randint(1, 20)
        letter = randomLetter(lastletter)
        lastletter = letter
        
        text += length * letter
        
        if (i + length > size):
            length = size - i
            
        if length > 1:
            compressed += str(length) 
        compressed += letter
        
        i += length
    
    text = text[0:size]
    
    with open(f"text_{size}.txt", "w") as text_file:
        text_file.write(text)
        
    with open(f"compressed_{size}.txt", "w") as text_file:
        text_file.write(compressed)

