###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#




#*****************************************
# To format this code. I used python black
#*****************************************

import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!

    # dictionary to store no of times a particular parts of speech occurs
    partofspeechdict = {}
    # dictionary to store the occured words and frequency of parts of speech associated with that word
    # e.g no of times nouns occurs in the dataset
    wordsdict = {}
    # dictionary to store frequency of the parts of speech transitions
    # e.g no of times that (verb-> noun) occurs in the dataset 
    hmmdict = {}
    # dictionary to store the frequency of the transitions that begin with certain parts of speech
    # e.g no of times that (verb-> pos) occurs in the dataset where pos is any parts of speech
    pospstartdict = {}
    # dictionary to store the transition probabilities
    Hmm_Prob_Dict = {}
    # dictionary to store the probabilities of the individual parts of speech
    Pos_Prob_Dict = {}

    def grammerrules(self, word):
        n = len(word)
        if word[n - 4 :] == "less" or word[n - 4 :] == "able":
            return "adj"
        if (
            word[n - 4 :] == "ence"
            or word[n - 4 :] == "ment"
            or word[n - 4 :] == "ness"
            or word[n - 4 :] == "ship"
        ):
            return "noun"

        if (
            word[n - 3 :] == "ful"
            or word[n - 3 :] == "ous"
            or word[n - 3 :] == "ish"
            or word[n - 3 :] == "ive"
        ):
            return "adj"
        if word[n - 3 :] == "ist" or word[n - 3 :] == "ion" or word[n - 3 :] == "ity":
            return "noun"
        if word[n - 3 :] == "ate" or word[n - 3 :] == "ify" or word[n - 3 :] == "ize":
            return "verb"
        if word[n - 2 :] == "al" or word[n - 2 :] == "ic":
            return "adj"
        if word[n - 2 :] == "er" or word[n - 2 :] == "or" or word[n - 2 :] == "ar":
            return "noun"
        if word[n - 2 :] == "ed":
            return "verb"
        if word[n - 2 :] == "ly":
            return "adv"
        if word[:3] == "dis" or word[:3] == "mis":
            return "verb"
        if (
            word[:2] == "en"
            or word[:2] == "il"
            or word[:2] == "im"
            or word[:2] == "in"
            or word[:2] == "ir"
        ):
            return "adj"
        if word[:2] == "ob" or word[:2] == "op" or word[:2] == "re" or word[:2] == "un":
            return "verb"
        if word[:4] == "anti":
            return "adj"

        return max(self.Pos_Prob_Dict.items(), key=lambda x: x[1])[0]

    def posterior(self, model, sentence, label):
        if model == "Simple":
            sum1 = 0
            for i in range(len(sentence)):
                pos = label[i]
                word = sentence[i]
                if word in self.wordsdict:
                    result = max(
                        [
                            (
                                (
                                    math.log(
                                        (self.wordsdict[word][pos])
                                        / self.partofspeechdict[pos]
                                    )
                                )
                                + math.log((self.Pos_Prob_Dict[pos])),
                                pos,
                            )
                            for pos in self.wordsdict[word].keys()
                        ]
                    )
                    sum1 = sum1 + result[0]
                else:

                    PosByGrammerRules = self.grammerrules(word)
                    if PosByGrammerRules:
                        sum1 += math.log(self.Pos_Prob_Dict[PosByGrammerRules])
                    else:
                        result = max(self.Pos_Prob_Dict.items(), key=lambda x: x[1])
                        pos = result[0]
                        sum1 += math.log(self.Pos_Prob_Dict[pos])
            return sum1

            # return -999
        elif model == "HMM":
            sum1 = 0
            prev = ""
            for i in range(len(sentence)):
                pos = label[i]
                word = sentence[i]
                if word in self.wordsdict:
                    result = math.log(
                        self.Hmm_Prob_Dict.get((prev, pos), 1)
                    ) + math.log(
                        (
                            (self.wordsdict[word].get(pos, 1))
                            / self.partofspeechdict[pos]
                        )
                    )
                    sum1 = sum1 + result
                else:

                    PosByGrammerRules = self.grammerrules(word)
                    if PosByGrammerRules:
                        sum1 += math.log(
                            self.Pos_Prob_Dict[PosByGrammerRules]
                        ) + math.log(self.Hmm_Prob_Dict.get((prev, pos), 1))
                    else:
                        result = max(self.Pos_Prob_Dict.items(), key=lambda x: x[1])
                        pos = result[0]
                        sum1 += math.log(self.Pos_Prob_Dict[pos]) + math.log(
                            self.Hmm_Prob_Dict.get((prev, pos), 1)
                        )
                prev = pos
            return sum1
            return -999
        elif model == "Complex":
            sum1 = 0
            prev = ""
            prevprev = ""
            for i in range(len(sentence)):
                pos = label[i]
                word = sentence[i]
                if word in self.wordsdict:
                    result = (
                        math.log(self.Hmm_Prob_Dict.get((prev, pos), 1))
                        + math.log(self.Hmm_Prob_Dict.get((prevprev, prev), 1))
                        + math.log(
                            (
                                (self.wordsdict[word].get(pos, 1))
                                / self.partofspeechdict[pos]
                            )
                        )
                    )
                    sum1 = sum1 + result
                else:

                    PosByGrammerRules = self.grammerrules(word)
                    if PosByGrammerRules:
                        sum1 += (
                            math.log(self.Pos_Prob_Dict[PosByGrammerRules])
                            + math.log(self.Hmm_Prob_Dict.get((prev, pos), 1))
                            + math.log(self.Hmm_Prob_Dict.get((prevprev, prev), 1))
                        )
                    else:
                        result = max(self.Pos_Prob_Dict.items(), key=lambda x: x[1])
                        pos = result[0]
                        sum1 += (
                            math.log(self.Pos_Prob_Dict[pos])
                            + math.log(self.Hmm_Prob_Dict.get((prev, pos), 1))
                            + math.log(self.Hmm_Prob_Dict.get((prevprev, prev), 1))
                        )
                prevprev = prev
                prev = pos
            return sum1
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
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

        # Building HMM net

        for j in self.hmmdict:
            self.Hmm_Prob_Dict[j] = self.hmmdict[j] / self.pospstartdict[j[0]]
        # print(Hmm_Prob_Dict)

        # Building Probability dictionary for part of speech

        totalPosCount = sum(self.partofspeechdict.values())
        for Pos in self.partofspeechdict:
            # print(Pos,partofspeechdict[Pos]," -",totalPosCount)
            self.Pos_Prob_Dict[Pos] = self.partofspeechdict[Pos] / totalPosCount
        

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):

        results = []
        for word in sentence:
            # if the word is present in the traindata
            if word in self.wordsdict:
                # appending the pos which has highest probability associated with the word into the result
                result = max([ ( 
                    math.log((self.wordsdict[word][pos]) / self.partofspeechdict[pos])
                    + math.log(self.Pos_Prob_Dict[pos]),pos) for pos in self.wordsdict[word].keys()]
                     )
                results.append(result[1])
            # If we encounters a new word
            else:
                # trying to find the parts of speech of the word using grammer rules
                PosByGrammerRules = self.grammerrules(word)
                if PosByGrammerRules:

                    results.append(PosByGrammerRules)
                # If we can't find it using grammer rules then assigning the POS tag which has highest probability 
                else:
                    result = max(self.Pos_Prob_Dict.items(), key=lambda x: x[1])

                    results.append(result[0])
        return results

    def hmm_viterbi(self, sentence):

        # Recursive function to calculate veterbi at each stage and return the final result
        def viterbi(vit_array, sentence, position):
            if position == len(sentence):
                return max(vit_array)[1]

            word = sentence[position]
            if position == 0:
                if word in self.wordsdict:
                    # iterating through all possible Pos combinations for the word
                    for pos in self.wordsdict[word]:
                        if ("", pos) in self.Hmm_Prob_Dict:
                            v_value = math.log(
                                self.Hmm_Prob_Dict[("", pos)]
                            ) + math.log(
                                (
                                    (self.wordsdict[word][pos])
                                    / self.partofspeechdict[pos]
                                )
                            )
                            vit_array.append((v_value, [pos]))

                    # If No Sentence starts with all possible POS of the word
                    if vit_array == []:
                        v_value = max(
                            [
                                (
                                    (
                                        (self.wordsdict[word][pos])
                                        / self.partofspeechdict[pos]
                                    ),
                                    pos,
                                )
                                for pos in self.wordsdict[word]
                            ]
                        )
                        vit_array.append((v_value[0], [v_value[1]]))

                    return viterbi(vit_array, sentence, position + 1)

                # If that word is not Present in the dictionary
                else:
                    PosByGrammerRules = self.grammerrules(word)
                    if PosByGrammerRules:
                        v_value = math.log(self.Hmm_Prob_Dict[("", PosByGrammerRules)])
                        vit_array.append((v_value, [PosByGrammerRules]))
                    else:
                        v_value = max(
                            [
                                (self.Hmm_Prob_Dict[i], i[1])
                                for i in self.Hmm_Prob_Dict
                                if i[0] == ""
                            ]
                        )
                        vit_array.append((v_value[0], [v_value[1]]))

                    return viterbi(vit_array, sentence, position + 1)

            # For postion other than starting
            else:
                result = []
                if word in self.wordsdict:
                    # Iterating through all possible Pos Combinations for the word
                    for pos in self.wordsdict[word]:
                        (maxval, maxarr) = (-100000000, [])
                        # Seeing all possible Transitions
                        for comb in vit_array:
                            if (comb[1][-1], pos) in self.Hmm_Prob_Dict:
                                val = (
                                    comb[0]
                                    + math.log(self.Hmm_Prob_Dict[(comb[1][-1], pos)])
                                    + math.log(
                                        (
                                            (self.wordsdict[word][pos])
                                            / self.partofspeechdict[pos]
                                        )
                                    )
                                )

                                # if local value greater than global value
                                if val > maxval:
                                    (maxval, maxarr) = (val, comb[1].copy())
                        ##To Do # Check what happens if none of transitions are in the dictionary ##
                        maxarr.append(pos)
                        result.append((maxval, maxarr))

                    return viterbi(result, sentence, position + 1)

                # If the word is not present in train data or if we see a new word
                else:
                    PosByGrammerRules = self.grammerrules(word)
                    if PosByGrammerRules:
                        pos1 = PosByGrammerRules
                    else:
                        pos1 = max(self.Pos_Prob_Dict.items(), key=lambda x: x[1])[0]

                    for comb in vit_array:
                        # appending the POS with maximium transition probablity from the given node
                        newcomb = comb[0] + math.log(
                            self.Hmm_Prob_Dict[(comb[1][-1], pos1)]
                        )
                        comb[1].append(pos1)
                        result.append((newcomb, comb[1].copy()))

                    

                    return viterbi(result, sentence, position + 1)

        k = viterbi([], sentence, 0)

        return k

    def complex_mcmc(self, sentence):
        # return [ "noun" ] * len(sentence)
        allpos = self.partofspeechdict.keys()
        samples = []
        sample = ["noun"] * len(sentence)
        result = ["noun"] * len(sentence)

        # Iterating 150 to genrate various samples
        convergecount = 0
        for i in range(150):
            # iterating through each word in the sentence
            for wordindex in range(len(sentence)):
                maxprobdict = []
                # iterating through all possible parts of speech
                for pos in allpos:
                    sample[wordindex] = pos
                    gibbprob = self.calculatepost_Gibbs(sample, sentence)
                    maxprobdict.append((gibbprob, pos))

                sample[wordindex] = max(maxprobdict)[1]
            if i > 5:
                samples.append(sample)
            # if the previous sample and this sample are equal then increase convergecount
            if len(samples) != 0 and sample == samples[-1]:
                convergecount += 1

            # If the Model converge then break
            if convergecount > 3:

                break

        for i in range(len(sentence)):
            resultdict = {}
            for j in range(len(samples)):

                resultdict[samples[j][i]] = resultdict.get(samples[j][i], 0) + 1
            #Assigning the POS which occur more number of times into the result.
            result[i] = max(resultdict, key=lambda x: resultdict[x])

        # print(result)
        return result

    # Function to calculate the posterier probability of the sample generated by gibbs
    def calculatepost_Gibbs(self, sample, sentence):
        probsum = 0
        for i in range(len(sample)):
            if sentence[i] in self.wordsdict:
                emisionprob = math.log(
                    (
                        (self.wordsdict[sentence[i]].get(sample[i], 0.01))
                        / self.partofspeechdict[sample[i]]
                    )
                )
            else:
                PosByGrammerRules = self.grammerrules(sentence[i])
                if PosByGrammerRules:
                    if PosByGrammerRules == sample[i]:
                        emisionprob = 100
                    else:
                        emisionprob = 0.01
                else:
                    emisionprob = 0.01
            if i == 0:
                try:
                    transprob = math.log(self.Hmm_Prob_Dict(("", sample[i]), 0.1))
                except:
                    transprob = 0.01

                probsum = probsum + transprob + emisionprob
            if i == 1:
                try:
                    transprob = math.log(
                        self.Hmm_Prob_Dict.get((sample[i - 1], sample[i]), 0.1)
                    )
                except:
                    transprob = 0.01
                probsum = probsum + transprob + emisionprob
            else:
                try:
                    transprob = math.log(
                        self.Hmm_Prob_Dict.get((sample[i - 1], sample[i]), 0.1)
                    )
                except:
                    transprob = 0.01
                try:
                    transprob2 = math.log(
                        self.Hmm_Prob_Dict.get((sample[i - 2], sample[i - 1]), 0.1)
                    )
                except:
                    transprob2 = 0.01

                probsum = probsum + emisionprob + transprob + transprob2
        return probsum

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")
