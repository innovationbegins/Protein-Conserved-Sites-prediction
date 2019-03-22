from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction import FeatureHasher
#from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
##import itertools
from Bio import SeqIO

import tkinter
from tkinter import *
from tkinter import messagebox
import re
topp = tkinter.Tk()
#root=tkinter.Frame(top)
#second = tkinter.Tk(root)

# Code to add widgets will go here...
AALetter = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


#############################################################################################
def CalculateAAComposition(ProteinSequence):

    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round((float(ProteinSequence.count(i)) / LengthSequence)*100, 2)
    return Result

#20 features


#############################################################################################
def CalculateDipeptideComposition(ProteinSequence):

    #400 dipepdite features

    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        for j in AALetter:
            Dipeptide = i + j
            Result[Dipeptide] = round((float(ProteinSequence.count(Dipeptide)) / (LengthSequence - 1))*100, 2)
    return Result



#############################################################################################

def Getkmers():

    #8000 tripeptide name
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers



#############################################################################################
def GetSpectrumDict(proteinsequence):
  #8000 tripeptide features
     
    result = {}
    kmers = Getkmers()
    for i in kmers:
        result[i] = len(re.findall(i, proteinsequence)*100)
    return result



#############################################################################################
def CalculateAADipeptideComposition(ProteinSequence):

    #8420 all uni,bi,tri features

    result = {}
    result.update(CalculateAAComposition(ProteinSequence))
    result.update(CalculateDipeptideComposition(ProteinSequence))

    result.update(GetSpectrumDict(ProteinSequence))

    return result



#############################################################################################



#################################

def select_seq():
   
   pid=E1.get()
   for record in SeqIO.parse ( "sequence.fasta", "fasta" ): #record=list(SeqIO.parse ( "sequence.fasta", "fasta" ))
   #print ('Record description',record.description)
   #print ('Sequence ',record.seq)
    if pid in record.description:
     p=str(record.seq)
     CalculateAAComposition(p)
     CalculateDipeptideComposition(p)
     GetSpectrumDict(p)
     res = CalculateAADipeptideComposition(p )
     
     import pandas as pd
     df = pd.read_csv('420.csv')
     X = df.drop('labels', axis=1)
     X = X.drop('pid', axis=1)
     y = df['labels']
     y=MultiLabelBinarizer().fit_transform(y)
     #X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0,test_size=0.05)
     messagebox.showinfo( "your sequence against id is:", record.seq)




 
 #print(df['labels'])
 
 #print(df.head())



         
     
'''     
MNTDQQPYQGQTDYTQGPGNGQSQEQDYDQYGQPLYPSQADGYYDPNVAAGTEADMYGQQ
PPNESYDQDYTNGEYYGQPPNMAAQDGENFSDFSSYGPPGTPGYDSYGGQYTASQMSYGE
PNSSGTSTPIYGNYDPNAIAMALPNEPYPAWTADSQSPVSIEQIEDIFIDLTNRLGFQRD
SMRNMFDHFMVLLDSRSSRMSPDQALLSLHADYIGGDTANYKKWYFAAQLDMDDEIGFRN
MSLGKLSRKARKAKKKNKKAMEEANPEDTEETLNKIEGDNSLEAADFRWKAKMNQLSPLE
RVRHIALYLLCWGEANQVRFTAECLCFIYKCALDYLDSPLCQQRQEPMPEGDFLNRVITP
IYHFIRNQVYEIVDGRFVKRERDHNKIVGYDDLNQLFWYPEGIAKIVLEDGTKLIELPLE
ERYLRLGDVVWDDVFFKTYKETRTWLHLVTNFNRIWVMHISIFWMYFAYNSPTFYTHNYQ
QLVDNQPLAAYKWASCALGGTVASLIQIVATLCEWSFVPRKWAGAQHLSRRFWFLCIIFG
INLGPIIFVFAYDKDTVYSTAAHVVAAVMFFVAVATIIFFSIMPLGGLFTSYMKKSTRRY
VASQTFTAAFAPLHGLDRWMSYLVWVTVFAAKYSESYYFLVLSLRDPIRILSTTAMRCTG
EYWWGAVLCKVQPKIVLGLVIATDFILFFLDTYLWYIIVNTIFSVGKSFYLGISILTPWR
NIFTRLPKRIYSKILATTDMEIKYKPKVLISQVWNAIIISMYREHLLAIDHVQKLLYHQV
PSEIEGKRTLRAPTFFVSQDDNNFETEFFPRDSEAERRISFFAQSLSTPIPEPLPVDNMP
TFTVLTPHYAERILLSLREIIREDDQFSRVTLLEYLKQLHPVEWECFVKDTKILAEETAA
YEGNENEAEKEDALKSQIDDLPFYCIGFKSAAPEYTLRTRIWASLRSQTLYRTISGFMNY
SRAIKLLYRVENPEIVQMFGGNAEGLERELEKMARRKFKFLVSMQRLAKFKPHELENAEF
LLRAYPDLQIAYLDEEPPLTEGEEPRIYSALIDGHCEILDNGRRRPKFRVQLSGNPILGD
GKSDNQNHALIFYRGEYIQLIDANQDNYLEECLKIRSVLAEFEELNVEQVNPYAPGLRYE
EQTTNHPVAIVGAREYIFSENSGVLGDVAAGKEQTFGTLFARTLSQIGGKLHYGHPDFIN
ATFMTTRGGVSKAQKGLHLNEDIYAGMNAMLRGGRIKHCEYYQCGKGRDLGFGTILNFTT
KIGAGMGEQMLSREYYYLGTQLPVDRFLTFYYAHPGFHLNNLFIQLSLQMFMLTLVNLSS
LAHESIMCIYDRNKPKTDVLVPIGCYNFQPAVDWVRRYTLSIFIVFWIAFVPIVVQELIE
RGLWKATQRFFCHLLSLSPMFEVFAGQIYSSALLSDLAIGGARYISTGRGFATSRIPFSI
LYSRFAGSAIYMGARSMLMLLFGTVAHWQAPLLWFWASLSSLIFAPFVFNPHQFAWEDFF
LDYRDYIRWLSRGNNQYHRNSWIGYVRMSRARITGFKRKLVGDESEKAAGDASRAHRTNL
IMAEIIPCAIYAAGCFIAFTFINAQTGVKTTDDDRVNSVLRIIICTLAPIAVNLGVLFFC
MGMSCCSGPLFGMCCKKTGSVMAGIAHGVAVIVHIAFFIVMWVLESFNFVRMLIGVVTCI
QCQRLIFHCMTALMLTREFKNDHANTAFWTGKWYGKGMGYMAWTQPSRELTAKVIELSEF
AADFVLGHVILICQLPLIIIPKIDKFHSIMLFWLKPSRQIRPPIYSLKQTRLRKRMVKKY
CSLYFLVLAIFAGCIIGPAVASAKIHKHIGDSLDGVVHNLFQPINTTNNDTGSQMSTYQS
HYYTHTPSLKTWSTIK

'''
 


    
    
topp.geometry("500x500")


dep=Label(topp, text = "Deep Bind Protein Binding Site Predictor.")
dep.config(width=200)
dep.config(font=("Courier", 15))
dep.place(x=20,y=20)
dep.pack(padx = 0, pady = 0)

 
   
B=Button(topp, text = "predict", command=lambda:[select_seq(),feature_extraction()])
L1=Label(topp, text = "Enter protein Id:")



L1.pack( side = LEFT,padx = 40, pady = 40)

E1=Entry(topp, bd = 3)
E1.pack( side = RIGHT,padx = 40, pady = 40)


B.place(x = 200,y = 300)


r=Label(topp, text = "Your Prediction:")
r.place(x=40,y=350)
#r.pack(padx = 40, pady = 40)
#r.pack( side = LEFT,padx = 300, pady = 300)






topp.mainloop()

   










