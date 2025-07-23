import sys
import re
# Node class for the decision tree
import mlops_id3.node as node
import math
train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    """
    Paramter:
        p: probability of P+ (yes)
    """    
    p_neg = 1-p

    if ((p == 0) | (p==1)): # log(0) is undefined
        return 0 #entr(0) = 0, entr(1) = 0
    ent = -(p)*math.log(p, 2) - (p_neg)*math.log(p_neg, 2)
    return ent

def count_value(data, var):
    """
    Paramters:
        var: int [[data]...[data]]
    Uses the global varnames for acxessing var index
    """
    cnt = 0

    try:
        var_idx = varnames.index(var)
    except ValueError as err:
        print(f'Invalid variable: {var}')
        exit(err)
    for x in data:
        if x[var_idx] == 1:
            cnt = cnt + 1
    return cnt
# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    """
    example:
        14/30
            13/17
            1/13
        total = 30
        pxi = 17
        py = 14
        py_pxi = 13
        entropy = e(14/30) - (e(13/17) + e(1/13))
        entropy = e(py / total) - (e(py_pxi / pxi) + e(((py - py_pxi) / (total-pxi))
        
        14
    """

    p_before = py / total
    entr_before = entropy(p_before)
    
    if (total <= 0):
        return 0
    if (pxi == 0): # pure split
        return 0
    if (pxi == total):
        return 0
    """
    if ((py_pxi/pxi)==1):
        return 1
    """
    val1_p_after = py_pxi / pxi
    val2_p_after = (py - py_pxi) / (total-pxi)

    v1_e = entropy(val1_p_after)
    v2_e = entropy(val2_p_after)

    # add weights
    gain = entr_before - ((pxi / total)*v1_e +((total-pxi)/total)*v2_e)

    if (gain < 0):
        print(f'eror: gain = e({py} / {total}) - ( e({py_pxi} / {pxi}) + e({py - py_pxi} / {total-pxi}) )')
        print(f'error:     = {entr_before} - ({v1_e} + {v2_e})')
    return float(gain)


def partition_data(data, var):
    p1 = []
    p2 = []
    try:
        var_idx = varnames.index(var)
    except ValueError as err:
        print(f'Invalid variable: {var}')
    for x in data:
        label = x[var_idx]
        if label == 1:
            p1.append(x)
        elif label == 0:
            p2.append(x)
        else:
            print("Invalid data")
            raise ValueError
    return p1, p2

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

def choose_var(data, var_names):
    """
    Chooses attribute with best infogain out of the remaining attributes in var_names.
    If multiple attributes have the same max gain, I chose to pick
    the one that is calculation last. This is an arbituary choice,
    but I chose this method because it appears to be the same arbituary decision that
    the mushroom example tree makes.
    """
    max_gain = -math.inf
    chosen_var = None
    gain = max_gain
    class_label = varnames[-1]
    same_gain = []
    gains = []
    if (len(var_names) == 0):
        print(f'error: empty varnames')
    for idx, var in enumerate(var_names):
        # p1 = cou
        #use var names for now
        # py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
        # pxi : number of occurrences of x_i=1
        # py : number of ocurrences of y=1
        # total : total length of the data
        pxi = count_value(data, var)

        if ((pxi == 0)): # pure split, skip calculation
            gain = 0
        else:
            py = count_value(data, class_label)
            if (py <= 0):
                print("error: all classes have same value")
                exit()
            #partition data
            p1, p2 = partition_data(data, var)
            py_pxi = count_value(p1, class_label)
            gain = infogain(py_pxi,pxi, py,len(data))
        gains.append(gain)
        if gain > max_gain:
            max_gain = gain
            chosen_var = var
    for idx, variable in enumerate(var_names):
        if gains[idx] == max_gain:
            same_gain.append(variable)
        
    newstr = ' '.join(same_gain)

    if (len(same_gain) > 0):
        prev_var = chosen_var
        chosen_var = same_gain[-1]

    chosen_p1, chosen_p2 = partition_data(data, chosen_var)
    return chosen_var, chosen_p1, chosen_p2, max_gain
# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.


def build_tree(data, variable_names):
    """
    Parameters:
        data:   list of ints
        var_names: list of strings
     Return:
        recursive -> root of the final tree
   
    """
    # The position of the class label is the last element in the list.
    """
    Build Tree info:
    Base cases:
        Data is empty: This is the only base case where we set the leaf to a default value
        No more attributes: Return leaf with majority class
        All data is of one class: Return leaf with that class (majority)
        Data has length of one: Return Leaf with the last examples class. This case may be redundent.
    Gain threshold: There is no gain threshold, however I commented out code 
    that I can choose to use to implement a threshold.
    """

    default_value = 0
    var_names = variable_names.copy()
    if varnames[-1] in var_names: # if root, need to remove class label from possible var's to split on
        var_names.remove(varnames[-1])
    
    #print(f'build_tree: len(data): {len(data)}, len(var_names): {len(var_names)}\n')
    val1 = 0
    val0 = 0
    for i, row in enumerate(data):
        row_class = row[-1]
        if (row_class == 0):
            val0 = val0 + 1
        elif (row_class == 1):
            val1 = val1 + 1
        else:
            print("error, invalid class")
            return
    
    if (len(data) == 1):
        #print(f'One row left, setting val to {data[0][-1]}')
        return node.Leaf(varnames, data[0][-1])
    
    if (len(data) ==0):
        # all 
        #print("no more examples\n")
        return node.Leaf(varnames, default_value)

    if (( (val0 == 0) & (val1>0) ) | ( (val1 == 0) & (val0>0) )):
        # only one class left
        #print("only one class left\n")
        if (val0 == 0): # if no class=0, return class=1 leaf
            return node.Leaf(varnames, 1)
        else:
            return node.Leaf(varnames, 0)
    if (len(var_names) < 1):
        #no variables left to split on, need to find majority class
        #print("no variables left to split on\n")
        if (val0 > val1):
            return node.Leaf(varnames, 0)
        else:
            return node.Leaf(varnames, 1)

    #print("not returning leaf\n")
    chosen_var, p1, p2, gain = choose_var(data, var_names)

    # gain threshold
    """
    if (gain < 0.02):
        if (val0 > val1):
            return node.Leaf(varnames, 0)
        else:
          return node.Leaf(varnames, 1)
    """
        
    if (chosen_var == None):# no vars left to chose
        print("error: no var chosen, but len(varnmaes) > 0")
        exit()
   
    else: 
        newnames = var_names.copy()
        newnames.remove(chosen_var)

        left = build_tree(p2, newnames)
        right = build_tree(p1, newnames)

        this_root =  node.Split(varnames, varnames.index(chosen_var), left, right)
        return this_root


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)
    return varnames

def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)
 

if __name__ == "__main__":
    main(sys.argv[1:])
    #our data is deciding if a 
    # mushroom is poisonous
