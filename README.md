# Part-of-speech-tagging
In this problem we should mark every word in a sentence with its part of speech. We are given initial train dataset with label. We have to use that dataset to train our model so that it works perfectively of the test data. We are using 3 approachs here

    Simple(Bayesian formula)
    HMM(veterbi)
    Complex MCMC(Gibbs Sampling)

Training the model

We will train the model using train dataset prior to testing it on test data. We are using probabilitic approches using bayesian nets here. So We need to precompute the probabilities using the training data. The probabilities we need to compute are

    Prior Probability (e.g. probabilty of noun in the dataset)
    emmision probability(e.g. probability of word "dance" being a verb)
    Transition probability(e.g. probabilty of noun->verb etc) Using the train data we are calculating these probabilities as shown below

def train(self, data):

        for (sen, posp) in data:
            prev = ""
            for i in range(len(sen)):
                # Pushing into partofspeechdict
                if posp[i] in self.partofspeechdict:
                    self.partofspeechdict[posp[i]] += 1
                else:
                    self.partofspeechdict[posp[i]] = 1
                # Pushing into words dict
                if sen[i] in self.wordsdict:
                    if posp[i] in self.wordsdict[sen[i]]:
                        self.wordsdict[sen[i]][posp[i]] += 1
                    else:
                        self.wordsdict[sen[i]][posp[i]] = 1
                else:
                    self.wordsdict[sen[i]] = {posp[i]: 1}
                # Pushing transition probabilities into transitionsdict
                if (prev, posp[i]) in self.hmmdict:
                    self.hmmdict[(prev, posp[i])] += 1
                else:
                    self.hmmdict[(prev, posp[i])] = 1
                # Pushing starting words probability into pospstartdict
                if prev in self.pospstartdict:
                    self.pospstartdict[prev] += 1
                else:
                    self.pospstartdict[prev] = 1

                prev = posp[i]
        for j in self.hmmdict:
            self.Hmm_Prob_Dict[j] = self.hmmdict[j] / self.pospstartdict[j[0]]

        totalPosCount = sum(self.partofspeechdict.values())
        for Pos in self.partofspeechdict:
          
            self.Pos_Prob_Dict[Pos] = self.partofspeechdict[Pos] / totalPosCount
        

Simple Approach(Bayesian Formula)

In this Approach we assume that each is word is independent from one another. We apply bayesian formula on it. P(P/W)=P(W/P)P(P) e.g. P(noun/"dog")=P("Dog"/noun)*P(noun)

We calculated the emmision probability prior using train data. In this simple algo we iterate over word in the sentence. Assigned the word to all possible parts of speech it may have. Then we calculate the posterior probabilty using bayesian formula. We assign the word with Parts od speech which has maximium posterior probability.
Missing Words

The major problem in this 3 algorithm is missing words. If we encouter a word which we didn't see previsiously in the train data. The algorithms fails to process it. In order to deal with the missing words, I used Grammer Rules. If the word in not present in the Train data.I will assign the parts of speech to that word using grammer rules. If even grammer rules function fails to recognize it then I will just assign a part of speech which is more frequent in the train dataset. This is the small example of grammer rules I used

if word[n - 3 :] == "ist" or word[n - 3 :] == "ion" or word[n - 3 :] == "ity":
            return "noun"

HMM(veterbi)

In this approach, along with the emmision probabilities we will also consider transition probabilities.At each word we build HMM. We take the sum of log probabilities of emmision and transition. We then propagate the Part of speech which has the highest posterior probability. We do this until we reach the end of the sentence then I just return the path which has maximium posterior probability.

P(P0=p0,P1=p1,......Pn=pn/W1,W2,.......Wn)=P(P0=p0)π(P(Pt=pt/Pt-1=pt-1))π(P(Wt/Pt=pt))

If we apply log on both sides the multiplication will be sumation and it is easier to calulate. Using this posterier formula we build the HMM and finally return the path which has maximium posterier probability
Complex MCMC(Gibbs Sampling)

In this model along with the emision probability we also consider the transition probability of this state and previous state.The posterier is

P(P0=p0,P1=p1,......Pn=pn/W1,W2,.......Wn)=P(P0=p0)π(P(Pt=pt/Pt-1=pt-1,Pt-2=pt-2))π(P(Wt/Pt=pt))

But gibbs sampling was very diffent then veterbi.In gibbs sampling, We generate samples. we assign random tags to all the words then we take a single word. We iterate over all possible tags that can be assigned to that word and take take the tag whicjh has maximium posterior probability. But we shouldn't change the tag of other words. after we are done with one word we move to another word. We do this until we reach the end of the sentence.We push this into samples list. We will do this process over and over again to get no of samples. Atlast the model will converge giving the resonable answer. Then we assign the word to the parts of speech which occurs more number of times in the samples for that particular word. Since time I am only running for 150 times. But I am also keeping track whether if the model converges earlier. We it converges we will if break the process.

 # if the previous sample and this sample are equal then increase convergecount
            if len(samples) != 0 and sample == samples[-1]:
                convergecount += 1

            # If the Model converge then break
            if convergecount > 3:

                break

This way I was able to aviod uneccesary iterations and was able to increase the running time of the algorithm.
Accuarcy

I was able to increase the accuracy of the model by dealing the missing words with grammer rules.Final accuarcy of the model is


                 Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       94.29%               49.65%
            2. HMM:       96.11%               60.95%
        3. Complex:       94.22%               53.10%

