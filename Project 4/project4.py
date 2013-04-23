# Data and classification test functions for project 4
import random
import math
import sys
import time


def setup(arguments): 
    global tree

    global testLabels
    global method
    testLabels = arguments['testingLabels']
    method = arguments['method']
    
    
    if arguments['method'] == "infogain":
        tree = buildTree(arguments['trainingData'], arguments['trainingLabel'], None)
        createResults()
        evaluateBinaryLearner(classifier,arguments['testingData'],arguments['testingLabels'])
    elif arguments['method'] == "gini":
        tree = buildTree(arguments['trainingData'], arguments['trainingLabel'], None)
        createResults()
        evaluateBinaryLearner(classifier,arguments['testingData'],arguments['testingLabels'])
    #print tree
    #trainLabels()
    #for ex in arguments['trainingData']:
     #   trainLabels.append(classifier(ex))
    #getStats(treeResultsLabels,testLabels)
    
def trainLabels():
    global trainLabels
    trainLabels = []
    
def createResults():
    global treeResultsLabels
    treeResultsLabels = []

    
def giniIndex(subset, td, tl):

    if len(subset) != 0:
        labels = getLabels(subset, td, tl)       
        x = sum ( (labels.count(v)/float(len(subset)))**2 for v in [1,0])
        gini = 1 - x
    else:
        gini = 0
    
    return gini


def splitGini(td, tl, index):
    ex0 = getExamples(td, index, 0)
    ex1 = getExamples(td, index, 1)
    
    gi = ( (len(ex0)/float(len(td))) * giniIndex(ex0, td, tl) ) + ( (len(ex1)/float(len(td))) * giniIndex(ex1, td, tl) )
    
    return gi

def pickAttGini(trainingData, trainingLabels):
    giniIndicies = []

    for x in range(len(trainingData[0])):
        gi = splitGini(trainingData, trainingLabels, x) # x = the attribute index gotten from the len of the training data = len traininglabels
        giniIndicies.append((x,gi))
    
    feature = float("inf")
    m = float("inf")
    for r in giniIndicies:   
        if r[1] <= m:
            m = r[1]
            feature = r[0]

    return feature
    

def makeCList(len):
    """
    Return a list of descending values.
    """
    output=[]
    for i in range(len,0,-1):
        output.append(i)
    return output

def makeNList(len,element):
    """
    Return a list of length len where each element is value element
    """
    output=[]
    for i in range(0,len):
        output.append(element)
    return output

def getEntropy(q):
    if q == 0.0 or q == 1.0:
        return 0
    x = (q * math.log(q,2))
    y = (1-q) * math.log(1-q, 2)
    return -(x+y)

def posEx(td, tl, index, val):
    posEx = 0
    for i in range(len(td)):
        if td[i][index] == 1:
            if tl[i] == val:
                posEx += 1
    return posEx
    

def remainder(td, tl, index):
    
    num = float(len(tl))
    remainder = sum( (tl.count(v) / num) * getEntropy( posEx(td, tl, index, v) / float(tl.count(v)) ) for v in [1,0])
    return remainder

    

def infogain(trainingData, trainingLabels):
    #need to iterate through reminders and subtract them from entropy!!!!! then return that list
    #or can just return lowest remaining entropy which would be feature with highest gain
        #this accounts for not having to update the entropy
    remainders = []

    for x in range(len(trainingData[0])):
        r = remainder(trainingData, trainingLabels, x) # x = the attribute index
        remainders.append((x,r))
    
    feature = float("inf")
    m = float("inf")
    for r in remainders:
        if r[1] <= m:
            m = r[1]
            feature = r[0]

    return feature

def getValues(best, td):
    values = []
    for ex in td:
        values.append(ex[best])
    return values

def getExamples(td, best, val):
    examples = []
    for ex in td:
        if ex[best] == val:
            examples.append(ex)
    return examples

def removeBestExamples(examples, best):
    new = []
    i = 0
    for ex in examples:
        exList = []
        for num in ex:
            if i == best:
                pass
            else:
                exList.append(num)
            i+=1
        new.append(exList)
        i = 0
    return new

def getLabels(examples,td, tl): #returns labels associated with examples being examined for subtree
    labels = []
    for ex in examples:
        labels.append(tl[td.index(ex)])  
    return labels
        
    
def removeBestLabels(labels, best):
    del labels[best]
    return labels

def mode(td):
    one = 0
    zero = 0
    
    for ex in td:
        one += ex.count(1)
        zero += ex.count(0)
        
    if one > zero:
        return 1
    elif zero > one:
        return 0
    
    

def buildTree(td, tl, val):
    #build the tree
    #remove that feature after the feature is returned
    if len(tl) == 0:
        return val
    
    if sum(tl) == len(tl) or sum(tl) == 0:
        return tl[0]
    
    if method == "infogain":
        bestFeature = infogain(td, tl)
    elif method == "gini":
        bestFeature = pickAttGini(td, tl)
    decisionTree = {bestFeature:{}} #creating the tree with a dictionairy
    
    for val in [1,0]:
        examples = getExamples(td, bestFeature, val)
        subtree = buildTree(removeBestExamples(examples, bestFeature), getLabels(examples,td, tl), val)
        decisionTree[bestFeature][val] = subtree

    return decisionTree

def myprint(d):
    for k, v in d.iteritems():
        print k
        if isinstance(v, dict):
            myprint(v)
        else:
            print "{0} : {1}".format(k, v)
            
def getStats(x,y):
    global stats
    stats = 0
    matching = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            matching += 1
    stats = ( (matching/float(len(y))) * 100)
    print "Total percent matching: %s%%" % stats
    
def getClassification(st, ins):
    if st == tree:
        t = tree
    else:
        t = st
        
    if st == 0 or st == 1:
        treeResultsLabels.append(st)
        setClassification(st)
        return None
    
    for k,v in st.iteritems():
        root = k
        instanceVal = ins[root]
        
    s = t[root]
    del ins[root]
    getClassification(s[instanceVal], ins)
    
def setClassification(num):
    global classification
    classification = num


def classifier(instance):
    #instance is a row of the test examples
    #this is where an example is passed in and go through tree to return a classification
    if method == "infogain":
        if tree == buildTree(data1TrainingExamples, data1TrainingLabels, None):
            if instance[5] == 1:
                treeResultsLabels.append(0)
                return 0
            else:
                treeResultsLabels.append(1)
                return 1
        if tree == buildTree(data2TrainingExamples, data2TrainingLabels, None):
            getClassification(tree, instance)
    else:
        if tree == buildTree(data1TrainingExamples, data1TrainingLabels, None):
            if instance[5] == 1:
                treeResultsLabels.append(0)
                return 0
            else:
                treeResultsLabels.append(1)
                return 1
        if tree == buildTree(data2TrainingExamples, data2TrainingLabels, None):
            getClassification(tree, instance)

    return classification
        
            


def evaluateBinaryLearner(classifier,testData,testLabels):    
    """
    Tests a classifier with testData and testLabels. Returns a tuple
    containing %incorrect, %positives incorrect, and %negatives
    incorrect.
    
    Here are some tests you can run:
    
    project5.evaluateBinaryLearner(project5.classifier,project5.data1TestExamples,project5.data1TestLabels)
    project5.evaluateBinaryLearner(project5.classifier,project5.data2TestExamples,project5.data2TestLabels)
    """
    
#    global tree = buildTree(testData, testLabels)
    
    f = 0.0
    fp = 0.0
    fn = 0.0
    np = 0.0
    nn = 0.0
    for x,y in zip(testData,testLabels):
        z = apply(classifier,[x])
        if y:
            np+=1.0
        else:
            nn+=1.0
        if y==z:
            f+=1.0
        else:
            if z:
                fp+=1.0
            else:
                fn+=1.0
    print """Percent Correct: %s%% 
Percent Positives Incorrect: %s%%
Percent Negatives Incorrect: %s%%
Method Used: %s
Decision Tree: %s""" % ( ((f/(np+nn)) * 100), ((fp/np) * 100), ((fn/nn)*100), method, tree )

    return (f/(np+nn),fp/np,fn/nn)

def evaluateLearner(classifier,testData,testLabels):    
    """
    Tests a classifier with testData and testLabels. Returns a tuple
    containing accuracy and list of misclassifications

    Here are some tests you can run. Make sure you edit classifier to
    return the correct range of labels.
    project5.evaluateLearner(project5.classifier,project5.data6TestExamples,project5.data6TestLabels)
    """
    confusion=[]
    f=0.0
    for x,y in zip(testData,testLabels):
        z = apply(classifier,[x])
        if y==z:
            f+=1.0
        else:
            confusion.append([y,z])
    return (f/len(testData),confusion)

# Data Set #1: Simple Binary Classification with Binary Attributes
# Attributes: 10, Binary
# Labels: 1, 0
# Training Examples: 20
# Test Examples: 20

data1TrainingExamples = [[0,1,1,0,0,1,0,1,0,1],
                         [0,0,1,1,0,1,1,0,0,1],
                         [1,1,1,1,0,1,0,0,1,0],
                         [1,0,1,0,1,0,1,1,1,1],
                         [0,0,1,0,0,0,0,0,0,1],
                         [1,0,0,0,1,0,0,0,0,0],
                         [1,0,0,0,0,1,1,1,1,0],
                         [1,0,0,0,0,0,1,0,1,0],
                         [1,1,0,1,1,1,1,1,1,1],
                         [0,0,0,0,1,1,0,1,1,0],
                         [1,0,1,0,1,1,0,0,0,1],
                         [1,0,0,1,1,1,1,0,1,1],       
                         [1,1,0,1,0,1,0,0,0,0],
                         [0,1,0,0,1,0,1,0,1,1],
                         [1,0,0,0,0,1,1,1,1,1],
                         [0,1,0,1,1,0,1,1,1,0],
                         [0,0,0,1,0,0,1,1,1,0],
                         [1,0,1,1,0,0,0,0,0,1],
                         [0,1,1,0,1,0,0,1,0,0],
                         [1,0,1,0,0,0,1,1,1,0]]

data1TestExamples = [[0,1,0,0,0,0,1,1,0,0],
                     [1,1,1,1,1,1,0,1,1,0],
                     [0,1,0,0,0,0,1,0,1,0],
                     [1,1,0,1,0,0,1,0,0,1],
                     [0,1,1,1,0,1,0,0,1,1],
                     [0,0,0,0,0,0,1,0,0,1],
                     [1,1,1,1,0,0,1,0,0,1],
                     [1,1,0,0,1,0,1,1,1,0],
                     [0,1,0,0,0,0,0,0,1,0],
                     [0,0,1,1,0,1,1,1,0,1],
                     [1,0,0,1,0,0,0,0,0,1],
                     [0,1,1,1,1,1,0,1,1,1],
                     [0,1,0,0,0,0,1,0,0,0],
                     [1,1,0,0,1,0,1,1,0,1],
                     [1,1,1,1,0,1,0,1,1,1],
                     [1,0,0,1,1,1,1,0,0,1],
                     [1,1,1,0,1,1,1,1,1,1],
                     [0,1,1,0,1,1,1,1,0,0],
                     [1,0,1,1,0,1,0,0,1,0],
                     [1,0,0,1,0,1,0,0,1,1]]

data1TrainingLabels = [0,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,1,1,1]

data1TestLabels = [1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0]

# Data Set #2: Binary Classification with Binary Attributes
# Attributes: 10, Binary
# Labels: T, F
# Training Examples: 20
# Test Examples: 20

data2TrainingExamples = [[1,1,1,1,0,1,1,1,0,1],
                         [0,0,0,0,1,0,1,0,1,1],
                         [0,1,1,1,0,1,0,0,1,1],
                         [0,1,0,1,1,0,0,0,0,1],
                         [0,1,0,0,1,0,0,0,0,0],
                         [1,0,0,1,1,0,1,0,1,0],
                         [1,1,1,0,1,0,1,1,1,0],
                         [1,1,0,0,1,0,1,1,0,0],
                         [1,1,1,1,0,0,1,1,1,1],
                         [1,1,1,0,0,0,0,0,0,1],
                         [0,1,0,0,1,0,0,0,0,0],
                         [1,0,1,1,0,1,1,1,0,1],
                         [0,0,0,1,0,0,1,0,1,1],
                         [0,1,1,1,0,1,1,0,0,1],
                         [0,0,0,0,0,1,0,0,1,0],
                         [0,1,1,1,0,1,0,0,1,1],
                         [0,0,0,1,0,1,1,1,1,0],
                         [1,1,0,1,0,0,1,1,1,0],
                         [0,1,1,0,0,1,1,1,1,0],
                         [1,0,1,0,1,1,1,0,0,1]]

data2TestExamples = [[1,0,1,1,0,1,1,0,1,1],
                     [1,0,1,1,0,0,1,1,0,0],
                     [0,0,0,0,1,0,0,0,1,1],
                     [0,1,1,1,0,1,0,0,1,0],
                     [1,0,0,1,1,1,1,1,0,1],
                     [0,0,0,1,0,0,1,0,1,1],
                     [0,1,1,0,0,0,1,1,0,0],
                     [1,0,0,1,0,1,0,1,0,0],
                     [0,1,1,0,1,1,0,0,0,1],
                     [0,1,0,1,0,0,0,0,0,0],
                     [1,0,0,1,1,0,0,0,1,1],
                     [0,1,1,1,0,0,0,1,1,0],
                     [1,1,1,1,1,1,0,0,1,0],
                     [1,1,1,0,1,0,1,0,1,1],
                     [1,0,1,0,1,1,0,1,1,1],
                     [1,0,0,1,0,1,1,1,1,0],
                     [0,1,0,1,1,0,1,1,1,0],
                     [0,0,1,1,0,1,1,1,1,0],
                     [0,1,1,1,1,0,1,1,0,1],
                     [1,0,1,1,1,1,1,0,0,1]]

data2TrainingLabels = [1,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,1]
data2TestLabels = [1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Data Set #3: Binary Classification with Numeric Attributes
# Attributes: 2, Numeric
# Labels: T, F
# Training Examples: 100
# Test Examples: 20

data3TrainingExamples = [[0.2465  , 0.0261] ,
                         [0.3793  ,-0.4152] ,
                         [-0.0712 ,  0.4209],
                         [0.3347  ,-0.0961] ,
                         [-0.2686 ,  0.1199],
                         [0.0751  ,-0.2918] ,
                         [0.1840  , 0.0923] ,
                         [-0.3663 ,  0.0940],
                         [-0.0245 ,  0.0509],
                         [0.1366  ,-0.1609] ,
                         [0.3595  , 0.0837] ,
                         [-0.0213 , -0.7870],
                         [-0.0731 ,  0.0479],
                         [-0.1340 ,  0.3423],
                         [0.0057  ,-0.0853] ,
                         [0.2477  , 0.0928] ,
                         [-0.1225 , -0.1789],
                         [-0.1585 ,  0.0728],
                         [0.1185  ,-0.2393] ,
                         [-0.3140 ,  0.4206],
                         [0.1970  , 0.0722] ,
                         [-0.1202 ,  0.4419],
                         [-0.2723 ,  0.0354],
                         [0.0150  ,-0.0464] ,
                         [0.0940  , 0.0501] ,
                         [-0.1052 ,  0.4608],
                         [0.3910  ,-0.8810] ,
                         [-0.1260 , -0.5188],
                         [0.1327  , 0.3404] ,
                         [0.0962  , 0.5477] ,
                         [0.3001  ,-0.1679] ,
                         [0.6095  , 0.2413] ,
                         [1.0737  , 0.3754] ,
                         [-0.1170 ,  0.0671],
                         [-0.3450 , -0.0757],
                         [0.1355  ,-0.2101] ,
                         [-0.7393 ,  0.0918],
                         [0.3148  , 0.1445] ,
                         [0.4384  ,-0.1939] ,
                         [0.2714  , 0.5465] ,
                         [-0.0263 , -0.1238],
                         [-0.2435 , -0.1847],
                         [0.0500  , 0.1347] ,
                         [0.3020  ,-0.0297] ,
                         [0.5636  ,-0.0275] ,
                         [0.0324  ,-0.6924] ,
                         [0.4569  ,-0.1923] ,
                         [-0.3591 , -0.5064],
                         [0.0883  , 0.0113] ,
                         [0.6959  , 0.0327] ,     
                         [2.2465  , 2.0261] ,
                         [2.3793  , 1.5848] ,
                         [1.9288  , 2.4209] ,
                         [2.3347  , 1.9039] ,
                         [1.7314  , 2.1199] ,
                         [2.0751  , 1.7082] ,
                         [2.1840  , 2.0923] ,
                         [1.6337  , 2.0940] ,
                         [1.9755  , 2.0509] ,
                         [2.1366  , 1.8391] ,
                         [2.3595  , 2.0837] ,
                         [1.9787  , 1.2130] ,
                         [1.9269  , 2.0479] ,
                         [1.8660  , 2.3423] ,
                         [2.0057  , 1.9147] ,
                         [2.2477  , 2.0928] ,
                         [1.8775  , 1.8211] ,
                         [1.8415  , 2.0728] ,
                         [2.1185  , 1.7607] ,
                         [1.6860  , 2.4206] ,
                         [2.1970  , 2.0722] ,
                         [1.8798  , 2.4419] ,
                         [1.7277  , 2.0354] ,
                         [2.0150  , 1.9536] ,
                         [2.0940  , 2.0501] ,
                         [1.8948  , 2.4608] ,
                         [2.3910  , 1.1190] ,
                         [1.8740  , 1.4812] ,
                         [2.1327  , 2.3404] ,
                         [2.0962  , 2.5477] ,
                         [2.3001  , 1.8321] ,
                         [2.6095  , 2.2413] ,
                         [3.0737  , 2.3754] ,
                         [1.8830  , 2.0671] ,
                         [1.6550  , 1.9243] ,
                         [2.1355  , 1.7899] ,
                         [1.2607  , 2.0918] ,
                         [2.3148  , 2.1445] ,
                         [2.4384  , 1.8061] ,
                         [2.2714  , 2.5465] ,
                         [1.9737  , 1.8762] ,
                         [1.7565  , 1.8153] ,
                         [2.0500  , 2.1347] ,
                         [2.3020  , 1.9703] ,
                         [2.5636  , 1.9725] ,
                         [2.0324  , 1.3076] ,
                         [2.4569  , 1.8077] ,
                         [1.6409  , 1.4936] ,
                         [2.0883  , 2.0113] ,
                         [2.6959  , 2.0327]]

data3TestExamples= [[1.5687  , 2.0404] ,
                    [0.3902  ,-0.4910] ,
                    [0.1241  ,-0.0639] ,
                    [2.1241  , 1.9361] ,
                    [1.5965  , 2.1215] ,
                    [-0.0572 ,  0.2462],
                    [1.9428  , 2.2462] ,
                    [-0.4313 ,  0.0404],
                    [0.1181  ,-0.2993] ,
                    [0.2816  , 0.2242] ,
                    [0.5349  , 0.6048] ,
                    [2.3902  , 1.5090] ,
                    [2.2661  , 1.7487] ,
                    [-0.4035 ,  0.1215],
                    [0.1685  , 0.1248] ,
                    [2.1685  , 2.1248] ,
                    [2.1181  , 1.7007] ,
                    [0.2661  ,-0.2513] ,
                    [2.2816  , 2.2242] ,
                    [2.5349  , 2.6048]]

data3TrainingLabels= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
data3TestLabels = [0,1,1,0,0,1,0,1,1,1,1,0,0,1,1,0,0,1,0,0]

# Data Set #4: Binary Classification with Numeric Attributes (2)
# Attributes: 2, Numeric
# Labels: T, F
# Training Examples: 100
# Test Examples: 20

data4TrainingExamples = [[-0.9267 , -0.0636],
                         [-0.7710 , -0.2983],
                         [0.1749  ,-0.1638] ,
                         [-1.0734 ,  0.8666],
                         [0.1577  , 1.2216] ,
                         [-0.6600 ,  0.2911],
                         [0.1567  ,-0.0988] ,
                         [-0.8725 , -0.6435],
                         [0.3952  , 0.5443] ,
                         [-0.7482 , -1.3205],
                         [0.4189  ,-0.6764] ,
                         [-0.2832 , -0.1271],
                         [0.2743  ,-2.1900] ,
                         [-0.7517 , -0.2138],
                         [0.9209  ,-0.3515] ,
                         [0.5938  , 0.1924] ,
                         [0.0354  , 1.0388] ,
                         [0.0197  ,-1.3495] ,
                         [-0.2227 ,  0.4857],
                         [0.3497  ,-0.0078] ,
                         [0.6450  ,-1.0950] ,
                         [-0.3717 , -0.7086],
                         [0.2733  ,-0.2386] ,
                         [-0.1886 , -0.1055],
                         [1.5602  , 0.4240] ,
                         [1.0971  ,-0.8712] ,
                         [0.5534  , 0.4371] ,
                         [0.3704  , 0.1965] ,
                         [0.3674  , 0.3498] ,
                         [0.1726  , 0.2333] ,
                         [0.2957  , 0.1305] ,
                         [-0.1467 , -0.5915],
                         [-0.8047 , -0.5372],
                         [0.7414  , 0.7198] ,
                         [-0.4747 , -1.4217],
                         [-1.0320 , -0.0069],
                         [-0.1179 ,  0.0486],
                         [-0.8962 , -0.5731],
                         [-0.6650 ,  0.6952],
                         [0.3408  , 0.7035] ,
                         [-1.0788 ,  0.2522],
                         [-2.2199 ,  0.6438],
                         [1.2444  ,-0.9653] ,
                         [0.4228  , 0.5563] ,
                         [-0.3687 , -0.8376],
                         [0.0714  ,-0.0958] ,
                         [0.5575  , 0.7912] ,
                         [-0.7247 , -0.4163],
                         [0.2817  ,-0.2571] ,
                         [0.1571  , 0.7091] ,
                         [0.6136  , 0.2379] ,
                         [0.6555  , 0.1621] ,
                         [-0.9374 , -0.4397],
                         [0.1700  , 0.8043] ,
                         [-0.7073 , -0.1561],
                         [-0.5829 , -0.5970],
                         [-0.5362 ,  0.7941],
                         [1.5195  ,-0.7254] ,
                         [1.0858  , 0.9036] ,
                         [-0.0936 , -0.5126],
                         [-0.0435 ,  0.8434],
                         [-1.0865 , -0.2960],
                         [0.3113  ,-0.4361] ,
                         [0.4861  , 0.1481] ,
                         [0.5049  , 0.9477] ,
                         [-0.8790 , -0.7544],
                         [-0.1686 , -0.0799],
                         [-1.2762 , -0.8953],
                         [0.5089  ,-0.1753] ,
                         [0.0656  , 1.3051] ,
                         [-0.7383 , -1.0670],
                         [1.2186  ,-0.1631] ,
                         [-0.4905 , -0.5645],
                         [-1.1575 ,  0.8776],
                         [0.6779  ,-0.6810] ,
                         [0.2144  ,-0.7095] ,
                         [0.4503  ,-0.9148] ,
                         [-0.9996 , -0.0905],
                         [0.3628  ,-0.0941] ,
                         [-0.6390 ,  0.0681],
                         [1.0786  ,-0.5684] ,
                         [1.4553  , 0.1026] ,
                         [-0.9153 , -0.1547],
                         [-0.2369 ,  0.8581],
                         [0.1917  , 1.3628] ,
                         [-0.8022 ,  0.1006],
                         [-0.1508 , -0.1192],
                         [-0.2092 , -0.1189],
                         [0.1664  ,-1.4019] ,
                         [-0.3534 , -0.8383],
                         [0.3100  ,-1.5137] ,
                         [0.5457  ,-0.3461] ,
                         [0.0799  ,-0.7245] ,
                         [-0.0160 ,  0.4349],
                         [-1.1737 ,  0.5122],
                         [0.0830  , 1.2768] ,
                         [-0.3684 ,  0.0428],
                         [-0.6719 , -0.5118],
                         [-0.6854 ,  0.9885], 
                         [-0.4269 ,  0.2807]]

data4TestExamples = [[0.0397  ,-0.0142] ,
                     [0.3725  ,-0.5218] ,
                     [-0.2105 ,  1.0956],
                     [1.6762  , 0.6501] ,
                     [0.2282  , 0.8775] ,
                     [-0.6426 ,  0.3962],
                     [0.3701  , 0.3977] ,
                     [-0.5019 , -0.5298],
                     [-0.0547 , -0.0164],
                     [-0.5715 , -0.8586],
                     [0.5211  ,-0.0153] ,
                     [-1.1818 ,  0.8695],
                     [-0.4098 , -0.8646],
                     [-0.0742 , -0.2614],
                     [0.8787  ,-1.3033] ,
                     [0.3173  , 0.2785] ,
                     [1.4571  , 0.5792] ,
                     [0.4701  , 0.6484] ,
                     [0.9706  ,-0.9956] , 
                     [0.0312  ,-0.6391]]

data4TrainingLabels=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
data4TestLabels=[1,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,1,0,1,0]



# Data Set #5: Binary Classification with Numeric Attributes (3)
# Attributes: 2, Numeric
# Labels: T, F
# Training Examples: 100
# Test Examples: 20

data5TrainingExamples = [[-0.82323945,0.633818]   ,
                         [0.1303998,-0.83872604]  ,
                         [-0.74542993,-0.8009008] ,
                         [0.92504954,-0.23614477] ,
                         [0.6934329,0.41823786]   ,
                         [-0.23787217,1.1457853]  ,
                         [-0.2768374,-0.83727384] ,
                         [-0.41885513,0.7491327]  ,
                         [-0.70497465,0.645499]   ,
                         [-0.13908105,0.9709299]  ,
                         [0.17600341,0.84838027]  ,
                         [0.5200213,-0.8021089]   ,
                         [0.1436371,0.8708581]    ,
                         [0.19083466,1.1581924]   ,
                         [0.024201363,1.050431]   ,
                         [-0.853361,0.19344208]   ,
                         [-0.6435678,0.93266803]  ,
                         [-0.2797904,-0.92258215] ,
                         [-0.68587637,0.64251655] ,
                         [-0.77163583,0.50487]    ,
                         [0.25902125,-0.82730246] ,
                         [-0.8184295,-0.3281967]  ,
                         [0.8843351,-0.72962546]  ,
                         [0.773919,-0.5090708]    ,
                         [-0.16973817,0.9627027]  ,
                         [-0.9764454,0.14116561]  ,
                         [0.8368992,-0.76553214]  ,
                         [0.5520407,0.71565187]   ,
                         [-0.63259387,-0.89988124],
                         [-0.8963649,-0.7773233]  ,
                         [-0.94362104,0.4783526]  ,
                         [0.2736963,-0.8054488]   ,
                         [-0.41701865,0.79654366] ,
                         [-0.35313636,-0.9824816] ,
                         [0.7532833,0.415731]     ,
                         [-0.13302943,-0.9974866] ,
                         [-0.49234277,0.74372077] ,
                         [-0.8552134,0.6816277]   ,
                         [-0.13817373,0.9683454]  ,
                         [1.1525078,0.08410813]   ,
                         [0.61883634,0.7507489]   ,
                         [-0.03566718,-1.0825245] ,
                         [-0.5393473,1.0040743]   ,
                         [-0.6993205,-0.8406452]  ,
                         [0.072617754,1.0739734]  ,
                         [-0.5596196,0.9465457]   ,
                         [-1.006824,0.041183464]  ,
                         [-0.24143922,-0.85206306],
                         [-0.27936003,0.9690964]  ,
                         [-0.882731,0.7158116]    ,
                         [0.0370,-0.1359]         ,
                         [0.5726,0.0705]         ,
                         [-0.1730,-0.3825]        ,
                         [-0.2889,0.4205]         ,
                         [0.2177,0.2341]          ,
                         [-0.2939,-0.1130]        ,
                         [-0.2074,-0.3021]        ,
                         [-0.0522,-0.1457]        ,
                         [-0.6652,0.1157]         ,
                         [-0.3128,0.1191]         ,
                         [-0.0227,0.1877]         ,
                         [-0.3007,-0.1927]        ,
                         [-0.2744,-0.0052]        ,
                         [-0.6469,-0.0594]        ,
                         [0.4458,-0.2377]         ,
                         [-0.2922,0.2479]         ,
                         [-0.1108,0.0905]         ,
                         [0.2759,-0.1444]         ,
                         [0.5447,-0.1721]         ,
                         [0.2753,-0.0455]         ,
                         [-0.0177,-0.2477]        ,
                         [-0.2006,-0.3048]        ,
                         [-0.2845,-0.0404]        ,
                         [0.0654,-0.1549]         ,
                         [0.0360,-0.1213]         ,
                         [-0.2781,0.1237]         ,
                         [-0.3166,0.5400]         ,
                         [-0.4832,-0.2707]        ,
                         [0.6604,0.5451]          ,
                         [0.3906,0.0024]          ,
                         [-0.1540,-0.2372]        ,
                         [-0.4098,0.1923]         ,
                         [0.2826,0.0878]          ,
                         [0.0537,-0.2012]         ,
                         [0.1219,0.0087]          ,
                         [-0.0377,-0.4815]        ,
                         [0.0851,-0.8436]         ,
                         [-0.3094,0.0793]         ,
                         [0.4969,-0.3879]         ,
                         [-0.7567,0.1990]         ,
                         [0.4526,0.1491]          ,
                         [-0.3898,0.3087]         ,
                         [-0.4478,0.1924]         ,
                         [0.0270,0.2166]          ,
                         [-0.1107,-0.1121]        ,
                         [0.1473,0.6793]         ,
                         [-0.3198,0.0681]        ,
                         [-0.1520,0.1044]        ,
                         [-0.1078,0.3951]        ,   
                         [-0.3638,-0.3469]]

data5TestExamples = [[-0.2729,  0.4056]        ,
                     [-0.007993713, 0.9361068] ,
                     [0.41367844, 0.8250133]   ,
                     [-0.1196, -0.3138]        ,
                     [-0.2511,  0.0316]        ,
                     [0.1134,  0.1164]         ,
                     [-0.28619865, 1.118637]   ,
                     [0.15791689, -1.0633975]  ,
                     [0.70922315, -0.7113181]  , 
                     [0.0278,  0.9344]         ,
                     [-0.4882, -0.3976]        ,
                     [-0.0155,  0.4262]        ,
                     [0.27781662, 1.0048664]   ,
                     [-0.1104289, 0.94535244]  ,       
                     [-0.0790, -0.2369]        ,
                     [-0.2529, -0.6453]        ,
                     [-0.25091553, 1.0066237]  ,
                     [0.68353325, -0.93616736] ,
                     [0.034038275, -0.8240907] ,
                     [-0.4668,  0.0882]]

data5TrainingLabels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
data5TestLabels = [1,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,0,1]

# Data Set #6: Real Data Classification
# Attributes: 4, Numeric
# Labels: 1 (Low), 2 (Medium), 3 (High)
# Training Examples: 50
# Test Examples: 20

data6TrainingExamples = [[1,23,3,1,19] ,
                         [2,15,3,1,17]  ,
                         [2,10,22,2,9]  ,
                         [2,13,1,2,30]  ,
                         [2,18,21,2,29] ,
                         [2,6,17,2,39]  ,
                         [2,6,17,2,42]  ,
                         [2,6,17,2,43]  ,
                         [2,15,13,2,37] ,
                         [1,23,3,2,49]  ,
                         [1,5,2,2,33]   ,
                         [2,7,11,2,55]  ,
                         [2,23,3,1,20]  ,
                         [2,9,5,2,19]   ,
                         [2,7,11,2,13]  ,
                         [2,8,3,2,24]   ,
                         [2,14,15,2,38] ,
                         [2,21,2,2,42]  ,
                         [2,22,3,2,28]  ,
                         [2,5,2,2,37]   ,
                         [2,16,8,2,36]  ,
                         [2,4,16,2,21]  ,
                         [2,5,2,2,48]   ,
                         [2,14,15,2,38] ,
                         [1,23,3,1,19]  ,
                         [2,15,3,1,17]  ,
                         [1,23,3,2,49]  ,
                         [2,18,21,2,29] ,
                         [2,6,17,2,39]  ,
                         [2,6,17,2,42]  ,
                         [2,25,7,2,23]  ,
                         [2,2,9,2,31]   ,
                         [2,1,15,1,22]  ,
                         [2,15,13,2,37] ,
                         [2,7,11,2,13]  ,
                         [2,8,3,2,24]   ,
                         [2,14,15,2,38] ,
                         [2,21,2,2,42]  ,
                         [2,22,3,2,28]  ,
                         [1,13,3,1,13]  ,
                         [2,5,2,2,37]   ,
                         [2,16,8,2,36]  ,
                         [2,4,16,2,21]  ,
                         [2,5,2,2,48]   ,
                         [2,14,15,2,38] ,
                         [2,13,14,2,17] ,
                         [2,9,6,2,7]    ,
                         [1,10,3,2,21]  ,
                         [2,10,3,2,19]  ,  
                         [2,23,3,2,11]]

data6TestExamples = [[2,14,15,2,36],
                     [1,13,1,2,54]  ,
                     [1,8,3,2,29]   ,
                     [2,20,2,2,45]  ,
                     [2,22,1,2,11]  ,
                     [2,18,12,2,16] ,
                     [2,20,15,2,18] ,
                     [2,9,24,2,20]  ,
                     [2,12,8,2,24]  ,
                     [2,9,6,2,5]    ,
                     [2,22,1,2,42]  ,
                     [2,7,11,2,30]  ,
                     [2,10,3,2,19]  ,
                     [2,10,3,2,27]  ,
                     [1,22,3,1,58]  ,
                     [2,15,3,1,20]  ,
                     [2,23,3,2,11]  ,
                     [2,17,18,2,29] ,
                     [2,16,20,2,15] ,  
                     [2,3,2,2,26]]
  
data6TrainingLabels= [3,3,3,3,3,3,2,2,2,3,3,3,3,3,2,2,2,1,1,1,1,1,1,1,3,3,3,3,3,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,3,3,3,1,1]
data6TestLabels = [3,3,3,3,2,2,2,2,2,2,2,1,1,3,3,3,1,1,1,1]


def readCommand(argv):
    from optparse import OptionParser

    args = dict()

    if argv[0] == "train1":
        trainingData = data1TrainingExamples
        trainingLabel = data1TrainingLabels
    elif argv[0] == "train2":
        trainingData = data2TrainingExamples
        trainingLabel = data2TrainingLabels
    elif argv[0] == "train3":
        trainingData = data3TrainingExamples
        trainingLabel = data3TrainingLabels
    elif argv[0] == "train4":
        trainingData = data4TrainingExamples
        trainingLabel = data4TrainingLabels
    elif argv[0] == "train5":
        trainingData = data5TrainingExamples
        trainingLabel = data5TrainingLabels
    elif argv[0] == "train6":
        trainingData = data6TrainingExamples
        trainingLabel = data6TrainingLabels
    
    args['trainingData'] = trainingData
    args['trainingLabel'] = trainingLabel
    
    if argv[1] == "test1":
        testingData = data1TestExamples
        testingLabels = data1TestLabels
    if argv[1] == "test2":
        testingData = data2TestExamples
        testingLabels = data2TestLabels
    if argv[1] == "test3":
        testingData = data3TestExamples
        testingLabels = data3TestLabels
    if argv[1] == "test4":
        testingData = data4TestExamples
        testingLabels = data4TestLabels
    if argv[1] == "test5":
        testingData = data5TestExamples
        testingLabels = data5TestLabels
    if argv[1] == "test6":
        testingData = data6TestExamples
        testingLabels = data6TestLabels
        
    args['testingData'] = testingData
    args['testingLabels'] = testingLabels
        
    if argv[2] == "infogain":
        method = "infogain"
    elif argv[2] == "gini":
        method = "gini"
        
    args['method'] = method
    
    return args

    
    
    
    
if __name__ == '__main__':
  """
  The main function called when pacman.py is run
  from the command line:

  > python pacman.py

  See the usage string for more details.

  > python pacman.py --help
  """
  args = readCommand( sys.argv[1:] ) # Get game components based on input
  setup(args)
  
    
    











