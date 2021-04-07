from tkinter import*
import random
import csv
import time
import sys
import numpy as np
from collections import Counter

reviews=list()
labels=list()

# Let's tweak our network from before to model these phenomena
class SentimentNetwork:
    def __init__(self, reviews,labels,min_count = 10,polarity_cutoff = 0.1,hidden_nodes = 10, learning_rate = 0.1):
       
        np.random.seed(1)
    
        self.pre_process_data(reviews, polarity_cutoff, min_count)
        
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
        
        
    def pre_process_data(self,reviews, polarity_cutoff,min_count):
        
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if(labels[i] == 'POSITIVE'):
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term,cnt in list(total_counts.most_common()):
            if(cnt >= 50):
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word,ratio in pos_neg_ratios.most_common():
            if(ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                if(total_counts[word] > min_count):
                    if(word in pos_neg_ratios.keys()):
                        if((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word.lower())
                    else:
                        review_vocab.add(word.lower())
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1,input_nodes))
        self.layer_1 = np.zeros((1,hidden_nodes))
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            if word in self.word2index.keys():
                self.layer_0[0][self.word2index[word]] += 1

    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def train(self, training_reviews_raw, training_labels):
        
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            #review = training_reviews[i]
            label = training_labels[i]
            review = training_reviews_raw[i]
            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer
            self.update_input_layer(review.lower());
            #Hidden layer
            self.layer_1 = self.layer_0.dot(self.weights_0_1)
            #self.layer_1 *= 0
            #for index in review:
                #self.layer_1 += self.weights_0_1[index]
            
            # Output layer
            self.layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # Output error
            layer_2_error = self.layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(self.layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate
            
            #for index in review:
                #self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            if(self.layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            if(self.layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start + 0.001)
            
           # sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
        
    
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            reviews_per_second = i / float(time.time() - start + 0.001)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        
        # Input Layer
        self.update_input_layer(review.lower());

        # Hidden layer
        self.layer_1 *= 0
        self.layer_1 = self.layer_0.dot(self.weights_0_1)
        #unique_indices = set()
        #for word in review.lower().split(" "):
            #if word in self.word2index.keys():
                #unique_indices.add(self.word2index[word])
        #for index in unique_indices:
            #self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
    def getScore(self, review):
        
        # Input Layer
        self.update_input_layer(review.lower());

        # Hidden layer
        self.layer_1 *= 0
        self.layer_1 = self.layer_0.dot(self.weights_0_1)
        #unique_indices = set()
        #for word in review.lower().split(" "):
            #if word in self.word2index.keys():
                #unique_indices.add(self.word2index[word])
        #for index in unique_indices:
            #self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        
        return layer_2[0]



def initialiseNeuralNetwork():
    g = open('reviews.txt','r') # What we know!
    global reviews
    reviews = list(map(lambda x:x[:-1],g.readlines()))
    g.close()

    g = open('labels.txt','r') # What we WANT to know!
    global labels
    labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
    g.close()
    
    mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.5,learning_rate=0.01)
    mlp.train(reviews[:-1000],labels[:-1000])
    return mlp

def getScoreForFeedback(feedback,mlp):
    return mlp.getScore(feedback)   




##################################################################
#global variable
matrix=[]
maxRows=0
testUser=[]
r=0
intr=0
feedback=""


def but1(x,tbr):
    global r
    global intr
    intr=1
    r=tbr
    x.destroy()

def showRecommendations(l):
    intr=0
    root = Tk()
    root.geometry("500x600+0+0")
    root.title("Recommendations")
    frame_stream = Frame(root)
    lbl_stream6 = Label(frame_stream, text = "These are Some Recommendations! Chose anyone!" , )
    lbl_stream = Label(frame_stream, text = l[0]  )
    lbl_stream1 = Label(frame_stream, text = l[1]  )
    lbl_stream2 = Label(frame_stream, text = l[2]  )
    lbl_stream3 = Label(frame_stream, text = l[3]  )
    lbl_stream4 = Label(frame_stream, text = l[4]  )
    lbl_stream5 = Label(frame_stream, text = l[5]  )

    b1 = Button(frame_stream,text="Chose")
    b2 = Button(frame_stream,text="Chose")
    b3 = Button(frame_stream,text="Chose")
    b4 = Button(frame_stream,text="Chose")
    b5 = Button(frame_stream,text="Chose")
    b6 = Button(frame_stream,text="Chose")

    b1.grid(row = 5, column = 10)
    b2.grid(row = 6, column = 10)
    b3.grid(row = 7, column = 10)
    b4.grid(row = 8, column = 10)
    b5.grid(row = 9, column = 10)
    b6.grid(row = 10, column = 10)

    b1.configure(command=lambda :but1(root,0))
    b2.configure(command=lambda :but1(root,1))
    b3.configure(command=lambda :but1(root,2))
    b4.configure(command=lambda :but1(root,3))
    b5.configure(command=lambda :but1(root,4))
    b6.configure(command=lambda :but1(root,5))

    lbl_stream.grid(row = 5,column = 0)
    lbl_stream1.grid(row = 6,column = 0)
    lbl_stream2.grid(row = 7,column = 0)
    lbl_stream3.grid(row = 8,column = 0)
    lbl_stream4.grid(row = 9,column = 0)
    lbl_stream5.grid(row = 10,column = 0)
    lbl_stream6.grid(row = 4,column = 0)



    frame_stream.pack()
    root.mainloop()


##########################################################

 
def showAllAvailableOptions(l):
    intr=0
    root = Tk()
    root.geometry("500x600+0+0")
    root.title("All Options Available")

    frame_stream = Frame(root)
    lbl_stream = Label(frame_stream, text = "These are Some Options Available! Chose one you like!" , )
    lbl_stream1 = Label(frame_stream, text = l[0]  )
    lbl_stream2 = Label(frame_stream, text = l[1]  )
    lbl_stream3 = Label(frame_stream, text = l[2]  )
    lbl_stream4 = Label(frame_stream, text = l[3]  )
    lbl_stream5 = Label(frame_stream, text = l[4]  )
    lbl_stream6 = Label(frame_stream, text = l[5]  )
    lbl_stream7 = Label(frame_stream, text = l[6]  )
    lbl_stream8= Label(frame_stream, text = l[7]  )
    lbl_stream9 = Label(frame_stream, text = l[8] )
    lbl_stream10 = Label(frame_stream, text = l[9]  )
    lbl_stream11= Label(frame_stream, text = l[10]  )
    lbl_stream12 = Label(frame_stream, text = l[11]  )
    lbl_stream13 = Label(frame_stream, text = l[12] )
    lbl_stream14 = Label(frame_stream, text = l[13]  )
    lbl_stream15 = Label(frame_stream, text = l[14]  )
    lbl_stream16 = Label(frame_stream, text = l[15]  )
    lbl_stream17 = Label(frame_stream, text = l[16]  )
  

    b1 = Button(frame_stream,text="Chose")
    b2 = Button(frame_stream,text="Chose")
    b3 = Button(frame_stream,text="Chose")
    b4 = Button(frame_stream,text="Chose")
    b5 = Button(frame_stream,text="Chose")
    b6 = Button(frame_stream,text="Chose")
    b7 = Button(frame_stream,text="Chose")
    b8 = Button(frame_stream,text="Chose")
    b9 = Button(frame_stream,text="Chose")
    b10 = Button(frame_stream,text="Chose")
    b11 = Button(frame_stream,text="Chose")
    b12= Button(frame_stream,text="Chose")
    b13 = Button(frame_stream,text="Chose")
    b14 = Button(frame_stream,text="Chose")
    b15 = Button(frame_stream,text="Chose")
    b16 = Button(frame_stream,text="Chose")
    b17 = Button(frame_stream,text="Chose")
   

    b1.grid(row = 6, column = 10)
    b2.grid(row = 7, column = 10)
    b3.grid(row = 8, column = 10)
    b4.grid(row = 9, column = 10)
    b5.grid(row = 10, column = 10)
    b6.grid(row = 11, column = 10)
    b7.grid(row = 12, column = 10)
    b8.grid(row = 13, column = 10)
    b9.grid(row = 14, column = 10)
    b10.grid(row = 15, column = 10)
    b11.grid(row = 16, column = 10)
    b12.grid(row = 17, column = 10)
    b13.grid(row = 18, column = 10)
    b14.grid(row = 19, column = 10)
    b15.grid(row = 20, column = 10)
    b16.grid(row = 21, column = 10)
    b17.grid(row = 22, column = 10)
  

    b1.configure(command= lambda :but1(root,0))
    b2.configure(command=lambda : but1(root,1))
    b3.configure(command=lambda : but1(root,2))
    b4.configure(command=lambda :but1(root,3))
    b5.configure(command= lambda :but1(root,4))
    b6.configure(command= lambda :but1(root,5))
    b7.configure(command=lambda : but1(root,6))
    b8.configure(command=lambda : but1(root,7))
    b9.configure(command=lambda : but1(root,8))
    b10.configure(command=lambda : but1(root,9))
    b11.configure(command=lambda : but1(root,10))
    b12.configure(command=lambda : but1(root,11))
    b13.configure(command=lambda : but1(root,12))
    b14.configure(command=lambda : but1(root,13))
    b15.configure(command=lambda : but1(root,14))
    b16.configure(command=lambda : but1(root,15))
    b17.configure(command=lambda : but1(root,16))


    lbl_stream.grid(row = 5,column = 0)
    lbl_stream1.grid(row = 6,column = 0)
    lbl_stream2.grid(row = 7,column = 0)
    lbl_stream3.grid(row = 8,column = 0)
    lbl_stream4.grid(row = 9,column = 0)
    lbl_stream5.grid(row = 10,column = 0)
    lbl_stream6.grid(row = 11,column = 0)
    lbl_stream7.grid(row = 12,column = 0)
    lbl_stream8.grid(row = 13,column = 0)
    lbl_stream9.grid(row = 14,column = 0)
    lbl_stream10.grid(row = 15,column = 0)
    lbl_stream11.grid(row = 16,column = 0)
    lbl_stream12.grid(row = 17,column = 0)
    lbl_stream13.grid(row = 18,column = 0)
    lbl_stream14.grid(row = 19,column = 0)
    lbl_stream15.grid(row = 20,column = 0)
    lbl_stream16.grid(row = 21,column = 0)
    lbl_stream17.grid(row = 22,column = 0)



    frame_stream.pack()

    root.mainloop()
def returnFeedback(x,string):
    global feedback
    global intr
    feedback=string
    x.destroy()
    intr=1
    
    
def getFeedback():
    global intr
    intr=0
    root = Tk()
    root.geometry("500x600+0+0")
    root.title("Give Feedback!!")
    frame= Frame(root)
    lbl=Label(frame,text="AFTER FOUR YEARS")
    lbl.pack(anchor="center")
   # txt=Text(frame)
    v14=StringVar()
    txt=Entry(frame,textvariable=v14,width=50)
    btn=Button(frame,text="Submit")
    btn.configure(command=lambda : returnFeedback(root,txt.get()))
    txt.pack()
    btn.pack()
    frame.pack()
    root.mainloop()




def writeToDataSet(a,field):
    fout=open("newDataset.txt","a+")
    v=[]
    for i in range(0,18):
        v.append(a[i])
    for i in range(18,36):
        v.append(0.00)
    v[field]=5.0
    for i in range(0,36):
        fout.write(str(v[i]))
        if(i!=35):
            fout.write(",")
    fout.write("\n")
    fout.close()

    
def opMinus(a,b):
    retVector=[]
    for i in range(0,len(a)):
        retVector.append((float)(a[i]-b))
    return retVector

def opMult(a,b):
    retVector=[]
    for i in range(0,len(a)):
        retVector.append((float)(a[i]*b[i]))
    return retVector

def sum(a):
    s=0.00
    for i in range(0,len(a)):
        s+=a[i]
    return s

def mean(a):
    return (float)(sum(a)/len(a))

def sqsum(a):
    s=0.00
    for i in range(0,len(a)):
        s+=pow(a[i],2)
    return s

def stdev(nums):
    n=len(nums)
    return pow(sqsum(nums)/n - pow(sum(nums)/n,2),0.5)

def pearsonCoeff(x,y):
    return sum( opMult(opMinus(x,mean(x)), opMinus(y,mean(y))) ) / len(x)*stdev(x)*stdev(y)

def setMatrix():
    global matrix
    global maxRows
    with open("newDataset.txt","r") as fin:
        reader=csv.reader(fin,dialect="excel",delimiter="\n")
        z=1
        for row in reader:
            y=[]
            x=(row[0].split(","))
            for i in range(0,len(x)):
                y.append((float)(x[i]))
            matrix.append(y)
            maxRows+=1
        fin.close()                       
    
        

# Main code below
def init(mlp):
 setMatrix()
 testUser[17]=5
 #names
 names=[
 "Maths", #0
 "Physics", #1
 "Chemistry", #2
 "Biology",#3
 "Accounts",#4
 "Business",#5
 "Economics",#6
 "Computer science",#7
 "Language",#8
 "social science",#9
 "Dance",#10
 "singing",#11
 "Drawing",#12
 "Drama and Acting",#13
 "science stream",#14
 "commerce stream",#15
 "arts stream",#16
     "",#17
 "Engineering",#18
 "B.Sc.",#19
 "MBBS",#20
 "BCA",#21
 "B.Arch.",#22
 "B.Pharmacy",#23
 "BBA",#24
 "B.Com",#25
 "LLB",#26
 "CA",#27
 "BBS",#28
 "BA",#29
 "BA + mathematics as additional",#30
 "Journalism and Mass communication",#31
 "Singing course",#32
 "Dance course",#33
 "Acting course",#34
 "Drawing artist"#35
 ];
     

 #sim
 sim=[] 
 global maxRows
 for i in range(0,maxRows):
      #newUser
      newUser=[]
      for j in range(0,17):
          newUser.append((float)(matrix[i][j]))
      ans=pearsonCoeff(newUser,testUser)
      sim.append(ans)

 #pred
 pred=[]

 sumD=0.00

 for i in range(0,len(sim)):
    sumD+=sim[i]

 for j in range(18,36):
    sumN=0.00
    for i in range(0,len(sim)):
        sumN+=sim[i]*matrix[i][j]
    rating=sumN/sumD
    pred.append((rating,j))
 pred=sorted(pred,key=lambda x:x[0],reverse=True)
 list_name=[]
 for i in range(0,5):
   list_name.append(names[pred[i][1]])
 list_name.append("Something else?")
##
 global r
 global intr
 showRecommendations(list_name)
 while(intr==0):
    print(r)
 index=r

 field=pred[index][1]
 #
 
 #
 if index==5:
     l=[]
     for i in range(18,36):
         l.append(names[i])
     showAllAvailableOptions(l)
     while(intr==0):
         print(r)
     index=r
     field=index+18
 getFeedback()
 while(intr==0):
     print(feedback)
 o=5*getScoreForFeedback(feedback,mlp);
 testUser[17]=o[0]
 writeToDataSet(testUser,field)
 return

 

    
def btn_submit_action_performed(x,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,mlp):
    print(v1.get())
    testUser[0]=float(v1.get())
    testUser[1]=float(v2.get())
    testUser[2]=float(v3.get())
    testUser[3]=float(v4.get())
    testUser[4]=float(v5.get())
    testUser[5]=float(v6.get())
    testUser[6]=float(v7.get())
    testUser[7]=float(v8.get())
    testUser[8]=float(v9.get())
    testUser[9]=float(v10.get())
    testUser[10]=float(v11.get())
    testUser[11]=float(v12.get())
    testUser[12]=float(v13.get())
    testUser[13]=float(v14.get())
    x.destroy()
    init(mlp)
                    
def btn_science_action_performed():
    testUser[14]=5
    testUser[15]=0
    testUser[16]=0
def btn_commerce_action_performed():
    testUser[14] = 0
    testUser[15] = 5
    testUser[16] = 0
def btn_arts_action_performed():
    testUser[14] = 0
    testUser[15] = 0
    testUser[16] = 5
    
    

def main():
    mlp=0
    mlp=initialiseNeuralNetwork()
    for i in range(0,18):
         testUser.append(0.00)
    testUser[17]=5
    root= Tk()
    root.geometry("500x600+0+0")
    root.title("Whatever")
    frame_stream=Frame(root)
    lbl_stream=Label(frame_stream,text="Choose your stream",anchor="w")
    btn_science=Button(frame_stream,text="Science",width=15)
    btn_commerce=Button(frame_stream,text="Commerce",width=15)
    btn_arts=Button(frame_stream,text="Arts",width=15)
    btn_science.configure(command=lambda :btn_science_action_performed())
    btn_commerce.configure(command=lambda :btn_commerce_action_performed())
    btn_arts.configure(command=lambda :btn_arts_action_performed())
    lbl_stream.grid(row=0,column=0,padx=10)
    btn_science.grid(row=0,column=1)
    btn_commerce.grid(row=1,column=1)
    btn_arts.grid(row=2,column=1)

    frame_input=Frame(root)
    lbl_sub1=Label(frame_input,text="Maths",anchor="w")
    lbl_sub2=Label(frame_input,text="Physics",anchor="w")
    lbl_sub3=Label(frame_input,text="Chemistry",anchor="w")
    lbl_sub4=Label(frame_input,text="Biology",anchor="w")
    lbl_sub5=Label(frame_input,text="Accounts",anchor="w")
    lbl_sub6=Label(frame_input,text="Business",anchor="w")
    lbl_sub7=Label(frame_input,text="Economics",anchor="w")
    lbl_sub8=Label(frame_input,text="Computer Science",anchor="w")
    lbl_sub9=Label(frame_input,text="Language",anchor="w")
    lbl_sub10=Label(frame_input,text="Social Science",anchor="w")
    lbl_sub11=Label(frame_input,text="Dance",anchor="w")
    lbl_sub12=Label(frame_input,text="Singing",anchor="w")
    lbl_sub13=Label(frame_input,text="Drawing",anchor="w")
    lbl_sub14=Label(frame_input,text="Drama and Acting",anchor="w")

    v1=StringVar()
    v2=StringVar()
    v3=StringVar()
    v4=StringVar()
    v5=StringVar()
    v6=StringVar()
    v7=StringVar()
    v8=StringVar()
    v9=StringVar()
    v10=StringVar()
    v11=StringVar()
    v12=StringVar()
    v13=StringVar()
    v14=StringVar()

    e1=Entry(frame_input,textvariable=v1)
    e2=Entry(frame_input,textvariable=v2)
    e3=Entry(frame_input,textvariable=v3)
    e4=Entry(frame_input,textvariable=v4)
    e5=Entry(frame_input,textvariable=v5)
    e6=Entry(frame_input,textvariable=v6)
    e7=Entry(frame_input,textvariable=v7)
    e8=Entry(frame_input,textvariable=v8)
    e9=Entry(frame_input,textvariable=v9)
    e10=Entry(frame_input,textvariable=v10)
    e11=Entry(frame_input,textvariable=v11)
    e12=Entry(frame_input,textvariable=v12)
    e13=Entry(frame_input,textvariable=v13)
    e14=Entry(frame_input,textvariable=v14)

    lbl_sub1.grid(row=0,column=0,padx=15)
    lbl_sub2.grid(row=1,column=0,padx=15)
    lbl_sub3.grid(row=2,column=0,padx=15)
    lbl_sub4.grid(row=3,column=0,padx=15)
    lbl_sub5.grid(row=4,column=0,padx=15)
    lbl_sub6.grid(row=5,column=0,padx=15)
    lbl_sub7.grid(row=6,column=0,padx=15)
    lbl_sub8.grid(row=7,column=0,padx=15)
    lbl_sub9.grid(row=8,column=0,padx=15)
    lbl_sub10.grid(row=9,column=0,padx=15)
    lbl_sub11.grid(row=10,column=0,padx=15)
    lbl_sub12.grid(row=11,column=0,padx=15)
    lbl_sub13.grid(row=12,column=0,padx=15)
    lbl_sub14.grid(row=13,column=0,padx=15)

    e1.grid(row=0,column=1)
    e2.grid(row=1,column=1)
    e3.grid(row=2,column=1)
    e4.grid(row=3,column=1)
    e5.grid(row=4,column=1)
    e6.grid(row=5,column=1)
    e7.grid(row=6,column=1)
    e8.grid(row=7,column=1)
    e9.grid(row=8,column=1)
    e10.grid(row=9,column=1)
    e11.grid(row=10,column=1)
    e12.grid(row=11,column=1)
    e13.grid(row=12,column=1)
    e14.grid(row=13,column=1)

    btn_submit=Button(frame_input,text="Submit")
    btn_submit.grid(row=14,column=1,pady=15)
    btn_submit.configure(command=lambda :btn_submit_action_performed(root,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,mlp))
    frame_stream.pack(pady=15)
    frame_input.pack()

    root.mainloop()
    
if __name__== "__main__":
  main() 
